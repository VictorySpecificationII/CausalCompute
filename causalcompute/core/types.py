"""Shared dataclasses for all pipeline steps. All quantities in SI units."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

RootComplex = Literal["single_socket", "dual_socket", "cascade"]


# ---------------------------------------------------------------------------
# Step 0 — Workload & hardware description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Workload:
    P: float          # parameters [count]
    Tok: float        # total training tokens [tokens]
    T: float          # wall-clock deadline [seconds]
    c: float = 6.0    # FLOPs per param-token (dense transformer ≈ 6)


@dataclass(frozen=True)
class StateBytes:
    """Single-copy model-state byte counts per parameter."""
    b_w: float = 2.0    # weight bytes/param   (bf16 → 2)
    b_g: float = 2.0    # gradient bytes/param (bf16 → 2)
    b_opt: float = 8.0  # optimizer bytes/param (Adam: m+v in fp32 → 8)


@dataclass(frozen=True)
class IO:
    b_tok: float = 2.0          # bytes/token in the dataset stream
    A_io: float = 1.3           # dataset BW headroom multiplier
    b_ckpt: float = 2.0         # bytes/param written per checkpoint
    t_ckpt_max: float = 300.0   # max tolerable checkpoint duration [s]


@dataclass(frozen=True)
class Device:
    """One accelerator, described in SI units."""
    F_dev_sust_flop_s: float   # sustained compute [FLOP/s]
    B_dev_mem_bytes: float     # on-device HBM capacity [bytes]


@dataclass(frozen=True)
class StepWorkingSet:
    """Instantaneous bytes that must be resident to execute one update step (global)."""
    B_step_bytes: float


@dataclass(frozen=True)
class StepSchedule:
    Tok_per_step: float  # tokens consumed per gradient-update step


@dataclass(frozen=True)
class AlgorithmStepFacts:
    b_update_per_param: float  # bytes of update signal per param per step
    k_update: float = 1.0      # margin multiplier (≥1)


@dataclass(frozen=True)
class FabricCapability:
    """Sustained per-node payload bandwidth for inter-node collectives [bytes/s]."""
    BW_node_sust_Bps: float


@dataclass(frozen=True)
class StorageCapability:
    """Sustained checkpoint write bandwidth [bytes/s]."""
    BW_ckpt_sust_Bps: float


@dataclass(frozen=True)
class CheckpointPolicy:
    seconds_per_ckpt: float  # wall-clock period between checkpoints [s]


# ---------------------------------------------------------------------------
# Step 1 — Design / parallelism knobs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignInputs:
    gpus_per_node: int = 8
    eta_compute: float = 0.35       # sustained compute efficiency ∈ (0, 1]
    eta_fabric: float = 0.80        # collective BW efficiency ∈ (0, 1]
    tp_max: int = 16                # max tensor-parallel degree to search
    pp_max: int = 16                # max pipeline-parallel degree to search
    g_max_multiplier: int = 8       # auto-size search ceiling = N_min × this
    comm_model: str = "ring_allreduce_dp_only"
    comm_exposed_fraction: float = 1.0  # 1.0 = no overlap, 0.0 = fully hidden
    zero_stage: int = 0             # ZeRO stage: 0 (full replica), 1, 2, or 3


# ---------------------------------------------------------------------------
# Step 2 — Power & thermals
# ---------------------------------------------------------------------------

CoolingMode = Literal["air", "liquid"]


@dataclass(frozen=True)
class PowerInputs:
    P_gpu_W: float = 700.0           # per-GPU TDP [W]
    P_cpu_W_per_node: float = 250.0  # CPU package(s) per node [W]
    P_other_W_per_node: float = 300.0  # DRAM + NICs + MB + fans + margin [W]
    PUE: float = 1.30                # power usage effectiveness (≥1)


@dataclass(frozen=True)
class AirCoolingInputs:
    deltaT_C: float = 15.0      # allowed air temperature rise [°C]
    rho_kg_m3: float = 1.20     # air density [kg/m³]
    cp_J_kgK: float = 1005.0    # specific heat capacity [J/(kg·K)]


@dataclass(frozen=True)
class LiquidCoolingInputs:
    deltaT_C: float = 7.0       # allowed coolant temperature rise [°C]
    rho_kg_m3: float = 1000.0   # coolant density [kg/m³]  (water-like)
    cp_J_kgK: float = 4186.0    # specific heat capacity [J/(kg·K)]


@dataclass(frozen=True)
class ThermalInputs:
    mode: CoolingMode = "air"
    air: AirCoolingInputs = field(default_factory=AirCoolingInputs)
    liquid: LiquidCoolingInputs = field(default_factory=LiquidCoolingInputs)


@dataclass(frozen=True)
class RackInputs:
    nodes_per_rack: Optional[int] = None
    rack_power_limit_W: Optional[float] = None
    racks: Optional[int] = None


# ---------------------------------------------------------------------------
# Step 3 — Network assumptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkInputs:
    """
    Leaf/spine fabric assumptions (vendor-agnostic).

    Defaults represent a rail-optimised InfiniBand NDR cluster:
      - 8 NICs per node (one per GPU, each on its own rail)
      - 64-port switches (NDR IB / 400GbE)
      - 400 Gb/s per port = 50 GB/s
      - 1:1 oversubscription (full bisection — standard for HPC)
      - Rail-optimised: each NIC connects to a dedicated leaf switch

    To model RoCE/Ethernet, change port_bw_Bps and set rail_optimised=False
    or nics_per_node to match your NIC count.
    """
    nics_per_node: int = 8            # NICs (and thus rails) per compute node
    switch_radix: int = 64            # total ports per switch
    port_bw_Bps: float = 5.0e10      # 400 Gb/s = 50 GB/s per port
    oversubscription: float = 1.0     # 1.0 = full bisection; 2.0 = 2:1
    rail_optimised: bool = True       # True = one NIC per GPU, independent rails


# ---------------------------------------------------------------------------
# Step 5 — Compute node specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComputeNodeSpec:
    """
    Physical compute node specification.

    Defaults represent an H100 SXM5 HGX node:
      - NVLink4 full-mesh via NVSwitches: 900 GB/s bisection BW per node
      - GPUDirect RDMA enabled (NIC DMAs directly to/from GPU HBM)
      - Dual-socket CPU: 128 logical cores (e.g. 2× AMD EPYC Genoa 64c)
      - 2 TB system DRAM

    Physics-relevant fields (affect consistency checks):
      intra_node_bw_Bps — if low (PCIe) and tp > 1, TP is bottlenecked
      gpudirect_rdma    — if False, fabric BW assumption in Step 0/3 may not hold

    BoM fields (informational):
      cpu_cores_per_node, dram_bytes_per_node, root_complex
    """
    intra_node_bw_Bps: float = 9.0e11      # NVLink4 bisection: 900 GB/s per node
                                            # PCIe 5.0 x16 ≈ 1.28e11 (128 GB/s, one-way)
    gpudirect_rdma: bool = True             # NIC → GPU HBM direct DMA
    cpu_cores_per_node: int = 128           # logical cores (e.g. 2× 64c EPYC Genoa)
    dram_bytes_per_node: float = 2.0e12    # system DRAM [bytes] (2 TB)
    root_complex: RootComplex = "dual_socket"  # PCIe root topology


# ---------------------------------------------------------------------------
# Step 4 — Storage design assumptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageInputs:
    """
    Storage cluster design assumptions (vendor-agnostic).

    Defaults represent a modern NVMe all-flash storage node:
      - 7 GB/s sequential per drive (NVMe SSD)
      - 7.68 TB per drive (enterprise NVMe)
      - 24 drives per storage node
      - 25 GB/s per storage node to the network
      - Dataset replicated once (no redundancy — compute cluster reads)
      - Checkpoints kept with 2× replication, 3 generations retained
    """
    drive_bw_seq_Bps: float = 7.0e9          # sequential read BW per drive [bytes/s]
    drive_capacity_bytes: float = 7.68e12    # usable capacity per drive [bytes]
    drives_per_storage_node: int = 24        # drives per storage node
    storage_net_bw_Bps: float = 2.5e10      # network BW per storage node [bytes/s]
    dataset_replication: int = 1             # dataset replica count
    ckpt_replication: int = 2               # checkpoint replica count
    ckpt_keep_count: int = 3               # rolling checkpoint generations to retain


# ---------------------------------------------------------------------------
# Step 6 — Cost model inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostInputs:
    """
    Unit costs and financial assumptions for the cost model.

    All prices in USD.  Defaults are ballpark circa-2024 market rates —
    adjust to current quotes before using for real procurement decisions.

    Component costs:
      gpu_unit_cost          H100 SXM5 list ~$30k; street varies widely
      node_chassis_cost      Compute node chassis + CPU(s) + DRAM, ex-GPU
      nic_unit_cost          Per NIC (NDR InfiniBand ~$2-3k; 400GbE ~$1-2k)
      switch_unit_cost       Per switch, leaf or spine (NDR 64-port ~$30-50k)
      cable_unit_cost        Per cable (AOC 2m ~$100; DAC cheaper)
      storage_node_cost      Per storage node chassis + CPU + DRAM, ex-drives
      drive_unit_cost        Per NVMe SSD (7.68TB enterprise ~$2-3k)
      rack_unit_cost         Per rack (cage + PDU + cabling infrastructure)

    OpEx:
      electricity_usd_kwh    US industrial average ~$0.07/kWh; DC contracts vary

    Amortization:
      capex_amortization_years  Typical HPC cluster: 3-5 years
    """
    # CapEx — hardware
    gpu_unit_cost: float = 30_000.0         # $ per GPU
    node_chassis_cost: float = 15_000.0     # $ per compute node (ex-GPU, ex-NIC)
    nic_unit_cost: float = 2_500.0          # $ per NIC
    switch_unit_cost: float = 40_000.0      # $ per switch (leaf or spine)
    cable_unit_cost: float = 100.0          # $ per cable
    storage_node_cost: float = 20_000.0     # $ per storage node (ex-drives)
    drive_unit_cost: float = 2_500.0        # $ per NVMe drive
    rack_unit_cost: float = 3_000.0         # $ per rack

    # OpEx
    electricity_usd_kwh: float = 0.07       # $ per kWh

    # Amortization
    capex_amortization_years: float = 3.0   # years over which CapEx is amortized


# ---------------------------------------------------------------------------
# Top-level Brief — single object passed through the whole pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Brief:
    # Step 0
    workload: Workload
    state: StateBytes
    io: IO
    device: Device
    step: StepWorkingSet
    schedule: StepSchedule
    update: AlgorithmStepFacts
    fabric: FabricCapability
    storage: StorageCapability
    checkpoint: CheckpointPolicy

    # Step 1
    design: DesignInputs
    design_G: Optional[int]  # None → auto-size

    # Step 2
    power: PowerInputs
    thermals: ThermalInputs
    rack: RackInputs

    # Step 3
    network: NetworkInputs = field(default_factory=NetworkInputs)

    # Step 4
    storage_inputs: StorageInputs = field(default_factory=StorageInputs)

    # Step 5
    node_spec: ComputeNodeSpec = field(default_factory=ComputeNodeSpec)

    # Step 6
    cost: CostInputs = field(default_factory=CostInputs)
