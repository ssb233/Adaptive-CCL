#include "pcie_backend.h"
#include "core/domain.h"

#ifdef AMPCCL_ENABLE_PCIE
#include "comm.hpp"
#include "ir.hpp"
#endif

#include <cstddef>
#include <vector>

namespace ampccl {

#ifdef AMPCCL_ENABLE_PCIE
namespace {

using namespace pccl;

// 2-rank AllReduce IR: Rank 0 root reduce, Rank 1 reduces into Rank 0's chunk.
IRProgram BuildAllReduceIR(int rank) {
    IRProgram program;
    program.input_chunk_count = 1;
    program.output_chunk_count = 1;

    if (rank == 0) {
        Instruction inst0;
        inst0.op = OpCode::D2H;
        inst0.src_numa = 0;
        inst0.src_chunk_idx = 0;
        inst0.dst_chunk_idx = 0;
        inst0.deps = {};
        inst0.effects = {{0}};

        Instruction inst1;
        inst1.op = OpCode::H2D;
        inst1.src_numa = 0;
        inst1.src_chunk_idx = 0;
        inst1.dst_chunk_idx = 0;
        inst1.deps = {{0, 0, 2}};
        inst1.effects = {};

        program.instructions = {inst0, inst1};
    } else {
        Instruction inst0;
        inst0.op = OpCode::D2H;
        inst0.src_numa = 0;
        inst0.src_chunk_idx = 0;
        inst0.dst_chunk_idx = 1;
        inst0.deps = {};
        inst0.effects = {{1}};

        Instruction inst1;
        inst1.op = OpCode::H2H_REDUCE;
        inst1.src_numa = 0;
        inst1.src_chunk_idx = 1;
        inst1.dst_chunk_idx = 0;
        inst1.deps = {{0, 0, 1}};
        inst1.effects = {{0}};

        Instruction inst2;
        inst2.op = OpCode::H2D;
        inst2.src_numa = 0;
        inst2.src_chunk_idx = 0;
        inst2.dst_chunk_idx = 0;
        inst2.deps = {{0, 0, 2}};
        inst2.effects = {};

        program.instructions = {inst0, inst1, inst2};
    }
    return program;
}

// 2-rank AllGather IR: each rank has 1 chunk input, 2 chunks output.
IRProgram BuildAllGatherIR(int rank) {
    IRProgram program;
    program.input_chunk_count = 1;
    program.output_chunk_count = 2;

    if (rank == 0) {
        Instruction inst0;
        inst0.op = OpCode::D2H;
        inst0.src_numa = 0;
        inst0.src_chunk_idx = 0;
        inst0.dst_chunk_idx = 0;
        inst0.deps = {};
        inst0.effects = {{0}};

        Instruction inst1;
        inst1.op = OpCode::D2D;
        inst1.src_numa = 0;
        inst1.src_chunk_idx = 0;
        inst1.dst_chunk_idx = 0;
        inst1.deps = {};
        inst1.effects = {};

        Instruction inst2;
        inst2.op = OpCode::H2D;
        inst2.src_numa = 0;
        inst2.src_chunk_idx = 1;
        inst2.dst_chunk_idx = 1;
        inst2.deps = {{0, 1, 1}};
        inst2.effects = {};

        program.instructions = {inst0, inst1, inst2};
    } else {
        Instruction inst0;
        inst0.op = OpCode::D2H;
        inst0.src_numa = 0;
        inst0.src_chunk_idx = 0;
        inst0.dst_chunk_idx = 1;
        inst0.deps = {};
        inst0.effects = {{1}};

        Instruction inst1;
        inst1.op = OpCode::D2D;
        inst1.src_numa = 0;
        inst1.src_chunk_idx = 0;
        inst1.dst_chunk_idx = 1;
        inst1.deps = {};
        inst1.effects = {};

        Instruction inst2;
        inst2.op = OpCode::H2D;
        inst2.src_numa = 0;
        inst2.src_chunk_idx = 0;
        inst2.dst_chunk_idx = 0;
        inst2.deps = {{0, 0, 1}};
        inst2.effects = {};

        program.instructions = {inst0, inst1, inst2};
    }
    return program;
}

}  // namespace
#endif

BackendResult BackendBase<PCIeBackend>::AllReduce(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int op,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    if (!domain || !domain->pcie_comm() || domain->pcie_nranks() != 2) {
        return BackendResult::Success;  // stub when no PCCL or not 2-rank
    }
    pcclComm_t comm = static_cast<pcclComm_t>(domain->pcie_comm());
    void* pcie_stream = domain->pcie_stream();
    if (!pcie_stream) {
        return BackendResult::UnhandledError;
    }
    int rank = domain->pcie_rank();
    pccl::IRProgram program = BuildAllReduceIR(rank);
    pcclResult_t ret = pcclSubmit(comm, program,
                                  const_cast<void*>(sendbuff), recvbuff,
                                  count, static_cast<pcclStream_t>(pcie_stream));
    return (ret == pcclSuccess) ? BackendResult::Success : BackendResult::UnhandledError;
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)datatype;
    (void)op;
    (void)stream;
    return BackendResult::Success;
#endif
}

BackendResult BackendBase<PCIeBackend>::AllGather(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    int datatype,
    void* stream) {
#ifdef AMPCCL_ENABLE_PCIE
    if (!domain || !domain->pcie_comm() || domain->pcie_nranks() != 2) {
        return BackendResult::Success;
    }
    pcclComm_t comm = static_cast<pcclComm_t>(domain->pcie_comm());
    void* pcie_stream = domain->pcie_stream();
    if (!pcie_stream) {
        return BackendResult::UnhandledError;
    }
    int rank = domain->pcie_rank();
    pccl::IRProgram program = BuildAllGatherIR(rank);
    pcclResult_t ret = pcclSubmit(comm, program,
                                  const_cast<void*>(sendbuff), recvbuff,
                                  sendcount, static_cast<pcclStream_t>(pcie_stream));
    return (ret == pcclSuccess) ? BackendResult::Success : BackendResult::UnhandledError;
#else
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)sendcount;
    (void)datatype;
    (void)stream;
    return BackendResult::Success;
#endif
}

BackendResult BackendBase<PCIeBackend>::ReduceScatter(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    void* stream) {
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)recvcount;
    (void)datatype;
    (void)op;
    (void)stream;
    return BackendResult::Success;
}

BackendResult BackendBase<PCIeBackend>::Broadcast(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int root,
    void* stream) {
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)datatype;
    (void)root;
    (void)stream;
    return BackendResult::Success;
}

}  // namespace ampccl
