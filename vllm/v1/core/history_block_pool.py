# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""page ownership history IDs and physical host-page bindings. Hot slots remain
static."""

from collections.abc import Iterable

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_block_hash
from vllm.v1.core.long_context_kv import HOST
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host.page_table import (
    HostPageTable,
    PageHandle,
    PageState,
    Result,
    SealTicket,
)
from vllm.v1.request import Request


class HistoryBlockPool(BlockPool):
    def configure_host(self, num_pages: int, num_workers: int, group_id: int) -> None:
        self.host_pages = HostPageTable(num_pages, num_workers)
        self.host_group_id = group_id
        self.hgen = [0] * self.num_gpu_blocks
        self.bindings: dict[int, PageHandle] = {}
        self.last_use: dict[tuple[int, int], int] = {}
        self.binding_updates: dict[int, int] = {}
        self.new_hashes: dict[str, list[tuple[int, int, bytes]]] = {}
        self.pending_seals: dict[str, dict[tuple[int, int, PageHandle], bytes]] = {}
        self.last_scheduled: dict[str, int] = {}
        self.step = 0
        self._jobs: dict[int, tuple[list[PageHandle], int]] = {}

    def _can_retire_binding(self, hid: int) -> bool:
        handle = self.bindings.get(hid)
        if handle is None:
            return False
        page = self.host_pages.get(handle)
        return (
            len(page.binds) == 1
            and not (page.leases or page.io or page.ticket)
            and self.host_pages.watermark
            >= max(page.last_use, self.last_use.get((hid, self.hgen[hid]), 0))
        )

    def get_num_free_blocks(self) -> int:
        if not hasattr(self, "host_pages"):
            return super().get_num_free_blocks()
        # page ownership conservatively charges each candidate, including ring scratch.
        # The normal free-queue order is retained. shared hot region replaces this bound
        # with
        # its source-aware four-domain admission contract.
        available = self.host_pages.available()
        count = 0
        block = self.free_block_queue.fake_free_list_head.next_free_block
        tail = self.free_block_queue.fake_free_list_tail
        while block is not None and block is not tail:
            available += int(self._can_retire_binding(block.block_id))
            if available <= 0:
                break
            count += 1
            available -= 1
            block = block.next_free_block
        return count

    def unbind(self, hid: int) -> None:
        handle = self.bindings.pop(hid, None)
        if handle is not None:
            hgen = self.hgen[hid]
            for request_id, pending in list(self.pending_seals.items()):
                pending.pop((hid, hgen, handle), None)
                if not pending:
                    del self.pending_seals[request_id]
            self.host_pages.unbind(handle, hid, hgen, self.last_use.pop((hid, hgen), 0))
            self.binding_updates[hid] = -1

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        blocks = super().get_new_blocks(num_blocks)
        scope = self.admission_observer._scope if self.admission_observer else ()
        for index, block in enumerate(blocks):
            hid = block.block_id
            self.unbind(hid)
            self.hgen[hid] += 1
            if scope is not None and HOST in scope:
                # host SSD persistence: a pre-eviction SSD write may only take pages
                # beyond the
                # rest of this admitted batch.
                handle = self.host_pages.reserve(pending=len(blocks) - index)
                assert handle is not None, "Host admission did not reserve enough pages"
                self.host_pages.bind(handle, hid, self.hgen[hid], self.step)
                self.bindings[hid] = handle
                self.binding_updates[hid] = handle.page_id
        return blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        super().cache_full_blocks(
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
            block_mask,
        )
        if kv_cache_group_id == self.host_group_id:
            entries = self.new_hashes.setdefault(request.request_id, [])
            for index in range(num_cached_blocks, num_full_blocks):
                block = blocks[index]
                if (
                    not block.is_null
                    and block.block_hash is not None
                    and self.host_pages.get(self.bindings[block.block_id]).state
                    == PageState.PRIVATE
                ):
                    entries.append(
                        (
                            block.block_id,
                            self.hgen[block.block_id],
                            request.block_hashes[
                                (index + 1) * (block_size // self.hash_block_size) - 1
                            ],
                        )
                    )
            # A receive may cache these blocks before admission. Preserve
            # their physical identities across steps until an actual producer
            # executes. Include pending local hits shared by another request.
            pending = self.pending_seals.setdefault(request.request_id, {})
            for index, block in enumerate(blocks[:num_full_blocks]):
                hid = block.block_id
                handle = self.bindings.get(hid)
                if handle is not None and block.block_hash is not None:
                    page = self.host_pages.get(handle)
                    if (
                        page.valid
                        and page.state == PageState.PRIVATE
                        and not page.ticket
                    ):
                        # Host keys use the terminal request-chain hash, while
                        # the GPU group hash concatenates all base-block hashes.
                        pending[hid, self.hgen[hid], handle] = request.block_hashes[
                            (index + 1) * (block_size // self.hash_block_size) - 1
                        ]

    def start_step(self, step: int) -> None:
        self.step = step
        self.new_hashes.clear()

    def cancel_scheduled(self, request_id: str) -> None:
        self.pending_seals.pop(request_id, None)
        for hid, hgen, _ in self.new_hashes.pop(request_id, []):
            if self.hgen[hid] == hgen:
                self.evict_blocks({hid})
                handle = self.bindings.get(hid)
                if handle is not None:
                    self.host_pages.get(handle).compute_result = Result.FAILED

    def finish_schedule(self, executed: Iterable[str]) -> list[SealTicket]:
        executed = set(executed)
        tickets = []
        for request_id in executed:
            self.last_scheduled[request_id] = self.step
            for (hid, hgen, handle), block_hash in self.pending_seals.pop(
                request_id, {}
            ).items():
                if self.hgen[hid] != hgen or self.bindings.get(hid) != handle:
                    continue
                page = self.host_pages.get(handle)
                if not page.valid or page.state != PageState.PRIVATE or page.ticket:
                    continue
                block = self.blocks[hid]
                if block.block_hash is None or not get_block_hash(
                    block.block_hash
                ).endswith(block_hash):
                    continue
                tickets.append(
                    self.host_pages.open_seal(
                        handle,
                        hid,
                        hgen,
                        make_offload_key(block_hash, self.host_group_id),
                        self.step,
                    )
                )
        return tickets

    def release_request(self, request_id: str, blocks: Iterable[KVCacheBlock]) -> None:
        self.pending_seals.pop(request_id, None)
        last_use = self.last_scheduled.pop(request_id, 0)
        for block in blocks:
            hid = block.block_id
            if hid in self.bindings:
                key = (hid, self.hgen[hid])
                self.last_use[key] = max(self.last_use.get(key, 0), last_use)
                page = self.host_pages.get(self.bindings[hid])
                page.last_use = max(page.last_use, last_use)

    def free_blocks(
        self, ordered_blocks: Iterable[KVCacheBlock], reuse_first: bool = False
    ) -> None:
        blocks = list(ordered_blocks)
        super().free_blocks(blocks, reuse_first)
        for block in blocks:
            hid = block.block_id
            if hid in self.bindings and block.ref_cnt == 0:
                if block.block_hash is None:
                    self.unbind(hid)
                else:
                    self.host_pages.get(self.bindings[hid]).binds[
                        hid, self.hgen[hid]
                    ] = "cached"

    def reset_prefix_cache(self) -> bool:
        # Check physical GPU references independently of the host capacity cap.
        if super().get_num_free_blocks() != self.num_gpu_blocks - 1:
            return False
        self.host_pages.reset()
        self.binding_updates.update({hid: -1 for hid in self.bindings})
        self.bindings.clear()
        self.last_use.clear()
        self.new_hashes.clear()
        self.pending_seals.clear()
        return super().reset_prefix_cache()

    def track_job(self, job_id: int, hids: list[int]) -> None:
        handles = list({self.bindings[hid] for hid in hids if hid in self.bindings})
        for handle in handles:
            self.host_pages.start_io(handle, job_id)
        self._jobs[job_id] = (handles, 0)

    def complete_jobs(self, completed: dict[int, int]) -> None:
        # Existing connector job counts retain their established contract.
        # Seal and step completion use rank masks independently.
        for job_id, count in completed.items():
            if job_id not in self._jobs:
                continue
            handles, seen = self._jobs[job_id]
            seen += count
            if seen < self.host_pages.num_workers:
                self._jobs[job_id] = (handles, seen)
                continue
            for handle in handles:
                self.host_pages.finish_io(handle, job_id)
            del self._jobs[job_id]
