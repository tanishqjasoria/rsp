//! Fetch + replay a mainnet block via RSP's BasicRpcDb and emit a zig-evm
//! bench-format RLP blob alongside RSP's bincode cache.
//!
//! This is a stripped copy of `rsp_host_executor::HostExecutor::execute` that
//! keeps the `BasicRpcDb` alive after execution so we can read `db.accounts`
//! (address-keyed) and `db.storage` (address+slot keyed) for the pre-state
//! section of our bench format. RSP's own cache drops those maps by the time
//! it's serialized.
//!
//! Output:
//!   - `<out>.zig.bin`: zig-evm bench blob, readable by `guest_main.zig`
//!   - `<out>.rsp.bin`: RSP's `ClientExecutorInput` (bincode) for the same block
//!
//! The two blobs represent the same block and pre-state, consumed by their
//! respective guests. Cycle counts from running each ELF on its blob are
//! directly comparable (same mainnet tx set, same pre-state witness work).

use std::{path::PathBuf, sync::Arc};

use alloy_consensus::{BlockHeader, TxReceipt};
use alloy_network::{BlockResponse, Ethereum};
use alloy_primitives::{Bloom, Sealable, B256, U256};
use alloy_provider::{Provider, ProviderBuilder};
use clap::Parser;
use eyre::{eyre, Context, Result};
use reth_chainspec::ChainSpec;
use reth_evm::{
    execute::{BasicBlockExecutor, Executor},
    ConfigureEvm,
};
use reth_evm_ethereum::EthEvmConfig;
use reth_primitives_traits::{Block as RethBlock, BlockBody, SealedHeader};
use reth_trie::{HashedPostState, KeccakKeyHasher};
use revm::database::CacheDB;
use revm_primitives::Address;
use rsp_client_executor::{
    custom::CustomEvmFactory, io::ClientExecutorInput, BlockValidator, IntoInput, IntoPrimitives,
    FromInput,
};
use rsp_primitives::genesis::Genesis;
use rsp_rpc_db::{BasicRpcDb, RpcDb};
use url::Url;

#[derive(Parser, Debug)]
#[command(name = "bench_from_rpc", about = "Fetch + replay a mainnet block and emit zig-evm + RSP blobs.")]
struct Args {
    #[arg(long)]
    block_number: u64,

    #[arg(long)]
    rpc_url: Url,

    /// Path prefix for output. `.zig.bin` and `.rsp.bin` are appended.
    #[arg(long)]
    out: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let block_number = args.block_number;

    let provider = ProviderBuilder::new().connect_http(args.rpc_url.clone());

    // Chain spec is mainnet. (Other chains could be plumbed in via --genesis-path.)
    let genesis = Genesis::Mainnet;
    let chain_spec: Arc<ChainSpec> = Arc::new((&genesis).try_into().unwrap());

    let evm_config = EthEvmConfig::new_with_evm_factory(
        chain_spec.clone(),
        CustomEvmFactory::new(None),
    );

    // Fetch current + parent blocks.
    tracing::info!("fetching block {block_number} and parent");
    let rpc_block = provider
        .get_block_by_number(block_number.into())
        .full()
        .await?
        .ok_or_else(|| eyre!("block {block_number} not found on RPC"))?;

    let current_block: reth_ethereum_primitives::Block =
        <reth_ethereum_primitives::EthPrimitives as IntoPrimitives<Ethereum>>::into_primitive_block(rpc_block.clone());

    let previous_block = provider
        .get_block_by_number((block_number - 1).into())
        .full()
        .await?
        .ok_or_else(|| eyre!("parent block {} not found on RPC", block_number - 1))?;
    let previous_block: reth_ethereum_primitives::Block =
        <reth_ethereum_primitives::EthPrimitives as IntoPrimitives<Ethereum>>::into_primitive_block(previous_block);

    // BasicRpcDb lives outside the execute call so we can read it afterward.
    let rpc_db = BasicRpcDb::<_, Ethereum>::new(
        provider.clone(),
        block_number - 1,
        previous_block.header().state_root(),
    );
    let accounts_ref = rpc_db.accounts.clone();
    let storage_ref = rpc_db.storage.clone();

    let cache_db = CacheDB::new(&rpc_db);
    let block_executor = BasicBlockExecutor::new(evm_config, cache_db);

    // Recover senders and validate header (same as HostExecutor).
    let block = current_block
        .clone()
        .try_into_recovered()
        .map_err(|_| eyre!("failed to recover senders"))?;

    <reth_ethereum_primitives::EthPrimitives as BlockValidator<ChainSpec>>::validate_header(
        &SealedHeader::seal_slow(
            <reth_ethereum_primitives::EthPrimitives as IntoPrimitives<Ethereum>>::into_consensus_header(rpc_block.header().clone()),
        ),
        chain_spec.clone(),
    )
    .wrap_err("header validation failed")?;

    let tx_count = current_block.body().transactions().count();
    tracing::info!("replaying block {block_number} with {tx_count} txs");
    let execution_output = block_executor.execute(&block).wrap_err("block execution failed")?;

    <reth_ethereum_primitives::EthPrimitives as BlockValidator<ChainSpec>>::validate_block_post_execution(
        &block,
        chain_spec.clone(),
        &execution_output,
    )
    .wrap_err("post-execution validation failed")?;

    // Build the (standard RSP) ClientExecutorInput for the .rsp.bin output.
    let mut logs_bloom = Bloom::default();
    execution_output.result.receipts.iter().for_each(|r| {
        logs_bloom.accrue_bloom(&r.bloom());
    });

    let state = rpc_db.state(&execution_output.state).await?;
    let bytecodes = rpc_db.bytecodes();
    let ancestor_headers = rpc_db.ancestor_headers().await?;

    let client_input: ClientExecutorInput<reth_ethereum_primitives::EthPrimitives> = ClientExecutorInput {
        current_block: <reth_ethereum_primitives::EthPrimitives as IntoInput>::into_input_block(current_block.clone()),
        ancestor_headers,
        parent_state: state,
        bytecodes: bytecodes.clone(),
        genesis: genesis.clone(),
        custom_beneficiary: None,
        opcode_tracking: false,
    };

    // Verify state root (same as HostExecutor does).
    let mut mutated = client_input.parent_state.clone();
    mutated.update(&HashedPostState::from_bundle_state::<KeccakKeyHasher>(
        &execution_output.state.state,
    ));
    let computed_root = mutated.state_root();
    if computed_root != current_block.header().state_root() {
        return Err(eyre!(
            "state root mismatch: computed={computed_root:?}, header={:?}",
            current_block.header().state_root()
        ));
    }
    tracing::info!("state root verified");

    // ---- Write RSP blob ----
    let rsp_path = args.out.with_extension("rsp.bin");
    let rsp_bytes = bincode::serialize(&client_input)?;
    std::fs::write(&rsp_path, &rsp_bytes)?;
    tracing::info!("wrote {} bytes to {}", rsp_bytes.len(), rsp_path.display());

    // ---- Build + write zig-evm blob ----
    // Collect every reachable node in the pre-state trie so zig-evm's
    // mpt_witness.verifyTrie can walk it against prev_state_root.
    let mut witness_nodes: Vec<(B256, Vec<u8>)> = Vec::new();
    collect_witness_nodes(&client_input.parent_state.state_trie, &mut witness_nodes);
    tracing::info!("collected {} witness nodes from state trie", witness_nodes.len());

    // Collect every reachable node across every touched account's
    // storage trie. zig-evm's mpt_update parses each touched account's
    // storage_root from this flat list. Accounts whose storage is
    // untouched keep their adopted storage_root and never parse.
    let mut storage_witness_nodes: Vec<(B256, Vec<u8>)> = Vec::new();
    for storage_trie in client_input.parent_state.storage_tries.values() {
        collect_witness_nodes(storage_trie, &mut storage_witness_nodes);
    }
    tracing::info!(
        "collected {} witness nodes across {} storage tries",
        storage_witness_nodes.len(),
        client_input.parent_state.storage_tries.len(),
    );

    let zig_bytes = build_zig_blob(
        &current_block,
        &previous_block,
        accounts_ref.read().unwrap().iter().map(|(a, i)| (*a, i.clone())).collect(),
        storage_ref.read().unwrap().iter().map(|(a, s)| (*a, s.clone().into_iter().collect::<Vec<_>>())).collect(),
        witness_nodes,
        storage_witness_nodes,
    )?;
    let zig_path = args.out.with_extension("zig.bin");
    std::fs::write(&zig_path, &zig_bytes)?;
    tracing::info!("wrote {} bytes to {}", zig_bytes.len(), zig_path.display());

    // ---- Sidecar: per-account post-state oracle for diffing zig-evm ----
    // RLP-encoded list of [addr, balance, nonce, code_hash, storage_list]
    // where storage_list = [[slot, value], ...]. Only touched accounts.
    let mut diff_entries: Vec<Vec<u8>> = Vec::new();
    for (addr, account) in execution_output.state.state.iter() {
        let info = match &account.info {
            Some(i) => i,
            None => continue, // self-destructed in this block; skip
        };
        let mut slots: Vec<Vec<u8>> = Vec::new();
        for (slot, slot_value) in account.storage.iter() {
            slots.push(rlp_encode_list(&[
                rlp_encode_u256(slot),
                rlp_encode_u256(&slot_value.present_value),
            ]));
        }
        diff_entries.push(rlp_encode_list(&[
            rlp_encode_bytes(addr.as_slice()),
            rlp_encode_u256(&info.balance),
            rlp_encode_u64(info.nonce),
            rlp_encode_bytes(info.code_hash.as_slice()),
            rlp_encode_list(&slots),
        ]));
    }
    let diff_rlp = rlp_encode_list(&diff_entries);
    let diff_path = args.out.with_extension("diff.bin");
    std::fs::write(&diff_path, &diff_rlp)?;
    tracing::info!(
        "wrote {} bytes ({} touched accounts) to {}",
        diff_rlp.len(),
        diff_entries.len(),
        diff_path.display()
    );

    // Also print a short summary.
    println!("block={block_number} txs={tx_count} pre_state_accounts={} storage_entries={}",
        accounts_ref.read().unwrap().len(),
        storage_ref.read().unwrap().values().map(|m| m.len()).sum::<usize>(),
    );
    println!("prev_state_root = 0x{}", hex::encode(previous_block.header().state_root()));
    println!("new_state_root  = 0x{}", hex::encode(current_block.header().state_root()));

    Ok(())
}

fn collect_witness_nodes(node: &rsp_mpt::MptNode, out: &mut Vec<(B256, Vec<u8>)>) {
    use alloy_rlp::Encodable;
    use rsp_mpt::MptNodeData;

    match node.as_data() {
        MptNodeData::Null | MptNodeData::Digest(_) => return,
        _ => {}
    }

    // Emit (hash, rlp-encoded) for this node.
    let mut buf = Vec::new();
    node.encode(&mut buf);
    // Only non-inline nodes (>= 32 bytes) are addressed by hash in the MPT.
    // Inline nodes are embedded in their parent's encoding and don't need a
    // separate witness entry — the walker (`mpt_witness.zig`) dispatches into
    // them directly when it sees an RLP list where a hash reference would be.
    if buf.len() >= 32 {
        let hash = node.hash();
        out.push((hash, buf));
    }

    // Recurse into children (skipping digest references which aren't in the
    // witness anyway).
    if let MptNodeData::Branch(children) = node.as_data() {
        for child in children.iter().flatten() {
            collect_witness_nodes(child, out);
        }
    } else if let MptNodeData::Extension(_, child) = node.as_data() {
        collect_witness_nodes(child, out);
    }
}

fn build_zig_blob(
    current_block: &reth_ethereum_primitives::Block,
    previous_block: &reth_ethereum_primitives::Block,
    accounts: Vec<(Address, revm_state::AccountInfo)>,
    storage: Vec<(Address, Vec<(U256, U256)>)>,
    witness_nodes: Vec<(B256, Vec<u8>)>,
    storage_witness_nodes: Vec<(B256, Vec<u8>)>,
) -> Result<Vec<u8>> {
    use alloy_rlp::Encodable;

    let header = current_block.header();

    // ---- block_input fields (11-item rlp list) ----
    let chain_id: U256 = U256::from(1u64); // mainnet
    let fork: u8 = fork_for_block(header.timestamp(), header.number());
    let parent_hash = previous_block.header().hash_slow();

    let block_input = rlp_encode_list(&[
        rlp_encode_u64(header.number()),
        rlp_encode_u64(header.timestamp()),
        rlp_encode_bytes(header.beneficiary().as_slice()),
        rlp_encode_u64(header.gas_limit()),
        rlp_encode_u256(&U256::from(header.base_fee_per_gas().unwrap_or(0))),
        rlp_encode_bytes(header.mix_hash().unwrap_or_default().as_slice()),
        rlp_encode_u256(&chain_id),
        rlp_encode_u64(header.excess_blob_gas().unwrap_or(0)),
        rlp_encode_bytes(header.parent_beacon_block_root().unwrap_or_default().as_slice()),
        rlp_encode_bytes(parent_hash.as_slice()),
        rlp_encode_u64(fork as u64),
    ]);

    // ---- pre_state: list of account entries ----
    let mut account_entries: Vec<Vec<u8>> = Vec::new();
    let mut skipped_empty = 0usize;
    for (addr, info) in accounts.iter() {
        // Reth's BasicRpcDb populates AccountInfo::default() (zero balance,
        // zero nonce, no code) for any address the EVM touches that doesn't
        // actually exist on chain. Those accounts have NO leaf in the state
        // trie — only a non-inclusion proof. The pre-state list must mirror
        // what the witness actually contains, otherwise the S3 cross-check
        // (`witness_crosscheck.zig::PreStateAccountMissingFromWitness`)
        // rejects the blob. Filter them out at the source.
        let code_bytes = info.code.as_ref().map(|c| c.original_bytes().to_vec()).unwrap_or_default();
        let is_empty = info.balance.is_zero()
            && info.nonce == 0
            && code_bytes.is_empty();
        if is_empty {
            skipped_empty += 1;
            continue;
        }

        // Collect this address's storage slots.
        let empty = Vec::new();
        let slots: &Vec<(U256, U256)> = storage
            .iter()
            .find(|(a, _)| a == addr)
            .map(|(_, s)| s)
            .unwrap_or(&empty);

        let mut storage_list = Vec::new();
        for (k, v) in slots.iter() {
            if v.is_zero() {
                continue;
            }
            storage_list.push(rlp_encode_list(&[
                rlp_encode_u256(k),
                rlp_encode_u256(v),
            ]));
        }

        account_entries.push(rlp_encode_list(&[
            rlp_encode_bytes(addr.as_slice()),
            rlp_encode_u256(&info.balance),
            rlp_encode_u64(info.nonce),
            rlp_encode_bytes(&code_bytes),
            rlp_encode_list(&storage_list),
        ]));
    }
    if skipped_empty > 0 {
        tracing::info!("skipped {skipped_empty} empty/non-existent accounts (no witness leaf)");
    }
    let pre_state = rlp_encode_list(&account_entries);

    // ---- raw_tx_list: each item is EIP-2718 tx envelope as rlp.bytes ----
    use alloy_eips::eip2718::Encodable2718;
    let mut raw_tx_items: Vec<Vec<u8>> = Vec::new();
    for tx in current_block.body().transactions() {
        let raw = tx.encoded_2718();
        raw_tx_items.push(rlp_encode_bytes(&raw));
    }
    let raw_tx_list = rlp_encode_list(&raw_tx_items);

    // ---- withdrawals ----
    let mut withdrawal_items: Vec<Vec<u8>> = Vec::new();
    if let Some(withdrawals) = current_block.body().withdrawals() {
        for w in withdrawals.iter() {
            withdrawal_items.push(rlp_encode_list(&[
                rlp_encode_u64(w.index),
                rlp_encode_u64(w.validator_index),
                rlp_encode_bytes(w.address.as_slice()),
                rlp_encode_u64(w.amount),
            ]));
        }
    }
    let withdrawals_rlp = rlp_encode_list(&withdrawal_items);

    // ---- roots ----
    let prev_state_root = previous_block.header().state_root();
    // Real-block post-state can't be recomputed from the partial witness the
    // guest holds, so we emit the all-zero sentinel to signal "skip".
    let expected_post_root = [0u8; 32];
    let prev_root_rlp = rlp_encode_bytes(prev_state_root.as_slice());
    let post_root_rlp = rlp_encode_bytes(&expected_post_root);

    // ---- pre-state trie witness: list of [node_hash, encoded_node] ----
    let witness_rlp = build_witness_list(&witness_nodes);

    // ---- storage-trie witness: flat list across every touched account ----
    let storage_witness_rlp = build_witness_list(&storage_witness_nodes);

    // ---- expected_header_rlp: canonical RLP of the committed header ----
    let mut header_buf = Vec::new();
    current_block.header().encode(&mut header_buf);
    let header_rlp = rlp_encode_bytes(&header_buf);

    // ---- parent_header_summary (v4): 6-item list ----
    // (gas_limit, base_fee, timestamp, gas_used, blob_gas_used,
    //  excess_blob_gas) — needed by the guest for EIP-1559 basefee
    // and EIP-4844 blob-basefee transition checks.
    let parent_hdr = previous_block.header();
    let parent_base_fee = U256::from(parent_hdr.base_fee_per_gas().unwrap_or(0));
    let parent_summary = rlp_encode_list(&[
        rlp_encode_u64(parent_hdr.gas_limit()),
        rlp_encode_u256(&parent_base_fee),
        rlp_encode_u64(parent_hdr.timestamp()),
        rlp_encode_u64(parent_hdr.gas_used()),
        rlp_encode_u64(parent_hdr.blob_gas_used().unwrap_or(0)),
        rlp_encode_u64(parent_hdr.excess_blob_gas().unwrap_or(0)),
    ]);

    // ---- outer list (v4, 10 items) ----
    let out = rlp_encode_list(&[
        block_input, pre_state, raw_tx_list, withdrawals_rlp,
        prev_root_rlp, post_root_rlp, witness_rlp,
        storage_witness_rlp, header_rlp, parent_summary,
    ]);
    Ok(out)
}

fn build_witness_list(nodes: &[(B256, Vec<u8>)]) -> Vec<u8> {
    let mut items: Vec<Vec<u8>> = Vec::new();
    for (hash, encoded) in nodes.iter() {
        items.push(rlp_encode_list(&[
            rlp_encode_bytes(hash.as_slice()),
            rlp_encode_bytes(encoded),
        ]));
    }
    rlp_encode_list(&items)
}

// ---------------------------------------------------------------------------
// Minimal RLP helpers. We use alloy_rlp's length/header primitives for byte
// strings; for lists we assemble manually since we already have encoded items.
// ---------------------------------------------------------------------------

fn rlp_encode_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    if bytes.len() == 1 && bytes[0] < 0x80 {
        out.push(bytes[0]);
        return out;
    }
    if bytes.len() <= 55 {
        out.push(0x80 + bytes.len() as u8);
        out.extend_from_slice(bytes);
        return out;
    }
    let len_bytes = minimal_be_bytes(bytes.len() as u64);
    out.push(0xb7 + len_bytes.len() as u8);
    out.extend_from_slice(&len_bytes);
    out.extend_from_slice(bytes);
    out
}

fn rlp_encode_list(items: &[Vec<u8>]) -> Vec<u8> {
    let total: usize = items.iter().map(|i| i.len()).sum();
    let mut out = Vec::new();
    if total <= 55 {
        out.push(0xc0 + total as u8);
    } else {
        let len_bytes = minimal_be_bytes(total as u64);
        out.push(0xf7 + len_bytes.len() as u8);
        out.extend_from_slice(&len_bytes);
    }
    for item in items {
        out.extend_from_slice(item);
    }
    out
}

fn rlp_encode_u64(v: u64) -> Vec<u8> {
    if v == 0 {
        return rlp_encode_bytes(&[]);
    }
    rlp_encode_bytes(&minimal_be_bytes(v))
}

fn rlp_encode_u256(v: &U256) -> Vec<u8> {
    let bytes: [u8; 32] = v.to_be_bytes();
    let start = bytes.iter().position(|&b| b != 0).unwrap_or(32);
    rlp_encode_bytes(&bytes[start..])
}

fn minimal_be_bytes(v: u64) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let start = bytes.iter().position(|&b| b != 0).unwrap_or(7);
    bytes[start..].to_vec()
}

/// Fork enum values from `zig-evm/src/frame.zig::Fork`:
///   Cancun = 0
///   Prague = 1
///   Osaka  = 2
///
/// Detection is timestamp-based (matches execution-specs ForkCriteria), not
/// block-number-based, because timestamps are stable across reorgs and match
/// what the EVM consensus layer actually keys off.
///
/// Mainnet activation timestamps:
///   Cancun: 1710338135 (2024-03-13 13:55:35 UTC)
///   Prague: 1746612311 (2025-05-07 10:05:11 UTC)
///   Osaka:  1764798551 (2025-12-03 21:49:11 UTC) — aka Fusaka
fn fork_for_block(timestamp: u64, number: u64) -> u8 {
    if timestamp >= 1_764_798_551 { 2 /* Osaka */ }
    else if timestamp >= 1_746_612_311 { 1 /* Prague */ }
    else if timestamp >= 1_710_338_135 { 0 /* Cancun */ }
    else { panic!("block {number} (ts={timestamp}) predates Cancun; zig-evm requires Cancun+") }
}
