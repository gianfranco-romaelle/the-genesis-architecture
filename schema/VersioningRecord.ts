/**
 * VersioningRecord — shared delta log for Object, HomΩ, and SuggestedAddition.
 *
 * Design: content-addressed linked list of JSON Patch (RFC 6902) deltas,
 * interfacing with Git via git_commit_sha / git_branch / git_repo_path.
 *
 * Tamper-detection: SHA-256 hashes of full serialized JSON at each state.
 * Rollback: rollback_patch is the inverse JSON Patch; applying it to the current
 *   state restores the previous state exactly (verifiable via previous_state_hash).
 */

/** RFC 6902 JSON Patch operation. */
export interface JsonPatchOperation {
  op: 'add' | 'remove' | 'replace' | 'move' | 'copy' | 'test';
  path: string;           // JSON Pointer (RFC 6901)
  value?: unknown;        // used by add, replace, test
  from?: string;          // used by move, copy
}

export type ChangedBy = 'llm' | 'script' | 'human';

export type ChangeType =
  | 'created'
  | 'updated'
  | 'merged'
  | 'split'
  | 'deprecated'
  | 'deleted';

export interface VersioningRecord {
  // ── Identity ─────────────────────────────────────────────────────────────
  record_id: string;             // UUID of this record
  object_version_id: string;     // UUID stamping the versioned object at this exact state

  // ── Pipeline tracing ──────────────────────────────────────────────────────
  pass_id: string;               // e.g., "W1-pass-003" or "normalize-run-007"
  timestamp: string;             // ISO 8601
  changed_by: ChangedBy;
  change_type: ChangeType;

  // ── Tamper-detection ──────────────────────────────────────────────────────
  // Both hashes are SHA-256 of the canonical JSON serialization (keys sorted,
  // no trailing whitespace) of the full object at that state.
  previous_state_hash: string | null;   // null only when change_type === 'created'
  current_state_hash: string;

  // ── Delta (forward + inverse) ─────────────────────────────────────────────
  // Applying delta_patch to the previous state produces the current state.
  // Applying rollback_patch to the current state restores the previous state.
  // rollback_patch is the computed inverse of delta_patch; verify with:
  //   sha256(apply(rollback_patch, current)) === previous_state_hash
  delta_patch: JsonPatchOperation[];
  rollback_patch: JsonPatchOperation[];

  // ── Rollback chain ────────────────────────────────────────────────────────
  // parent_delta_id forms a linked list: follow it to replay or roll back history.
  parent_delta_id: string | null;

  // ── Git interface ─────────────────────────────────────────────────────────
  // The pipeline commits each pass's changes to a git branch. These fields
  // bind each VersioningRecord to the authoritative git history, enabling
  // `git log`, `git diff`, `git revert`, and `git blame` on the graph state.
  git_commit_sha: string | null;    // 40-char SHA1 of the git commit
  git_branch: string | null;        // e.g., "pipeline/pass-003"
  git_repo_path: string | null;     // relative path of the serialized object file

  // ── Provenance ────────────────────────────────────────────────────────────
  commit_message: string;
  provenance_notes: string | null;
  affected_object_ids: string[];    // all Object IDs touched in the same operation
  affected_edge_ids: string[];      // all HomΩ edge IDs touched in the same operation
}
