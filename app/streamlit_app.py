"""
CausalCompute — Streamlit web UI.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from causalcompute.core.types import (
    AlgorithmStepFacts,
    AirCoolingInputs,
    CheckpointPolicy,
    ComputeNodeSpec,
    CostInputs,
    DesignInputs,
    Device,
    FabricCapability,
    IO,
    LiquidCoolingInputs,
    NetworkInputs,
    PowerInputs,
    RackInputs,
    StateBytes,
    StepSchedule,
    StepWorkingSet,
    StorageCapability,
    StorageInputs,
    ThermalInputs,
    Workload,
)
from causalcompute.core.fundamentals import run_fundamentals
from causalcompute.core.design import run_design
from causalcompute.core.thermals import run_thermals
from causalcompute.core.network import run_network
from causalcompute.core.storage import run_storage
from causalcompute.core.bom import run_bom
from causalcompute.core.cost import run_cost
from causalcompute.core.analysis import run_bottleneck


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CausalCompute",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ CausalCompute")
st.caption("Vendor-neutral, first-principles AI training infrastructure sizing")


# ---------------------------------------------------------------------------
# Sidebar — Brief parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Brief")

    st.subheader("Workload")
    P_B = st.number_input("Parameters (B)", value=13.0, min_value=0.1, step=1.0,
                           help="Model parameter count in billions")
    Tok_T = st.number_input("Tokens (T)", value=3.0, min_value=0.1, step=0.5,
                             help="Total training tokens in trillions")
    T_days = st.number_input("Deadline (days)", value=30.0, min_value=0.1, step=1.0)
    c = st.number_input("FLOPs / param-token", value=6.0, min_value=1.0, step=1.0,
                         help="~6 for dense transformers")

    st.subheader("Model State")
    b_w = st.number_input("Weight bytes/param", value=2.0, min_value=1.0, step=1.0,
                           help="2 = bf16/fp16, 4 = fp32")
    b_g = st.number_input("Gradient bytes/param", value=2.0, min_value=1.0, step=1.0)
    b_opt = st.number_input("Optimizer bytes/param", value=8.0, min_value=0.0, step=2.0,
                              help="Adam: 8 (m+v in fp32), SGD: 4")

    st.subheader("Device")
    F_dev_TFLOPS = st.number_input("Sustained compute (TFLOP/s)", value=1000.0, min_value=1.0, step=100.0,
                                    help="H100 SXM BF16 ≈ 1000 TFLOP/s peak; 350 TFLOP/s sustained")
    B_mem_GB = st.number_input("Device memory (GB)", value=80.0, min_value=1.0, step=16.0)

    st.subheader("Step & Schedule")
    B_step_GB = st.number_input("Step working set (GB)", value=150.0, min_value=1.0, step=10.0,
                                  help="Instantaneous activations + temps for one step (global)")
    Tok_per_step_M = st.number_input("Tokens per step (M)", value=40.0, min_value=0.1, step=1.0)

    st.subheader("Fabric & Storage")
    BW_fabric_GBs = st.number_input("Node fabric BW (GB/s)", value=100.0, min_value=1.0, step=25.0,
                                     help="Sustained per-node inter-node bandwidth")
    BW_ckpt_GBs = st.number_input("Checkpoint BW (GB/s)", value=5.0, min_value=0.1, step=1.0)
    ckpt_interval_h = st.number_input("Checkpoint interval (h)", value=1.0, min_value=0.1, step=0.5)

    st.subheader("Design (Step 1)")
    gpus_per_node = st.selectbox("GPUs per node", [1, 2, 4, 8, 16], index=3)
    eta_compute = st.slider("Compute efficiency η", 0.05, 1.0, 0.35, 0.05,
                             help="Sustained vs peak FLOP/s ratio")
    eta_fabric = st.slider("Fabric efficiency η", 0.10, 1.0, 0.80, 0.05,
                            help="Effective vs line-rate bandwidth ratio")
    fixed_G = st.number_input("Fixed GPU count (0 = auto-size)", value=0, min_value=0, step=8)
    design_G = None if fixed_G == 0 else int(fixed_G)

    st.subheader("Power & Thermals (Step 2)")
    P_gpu_W = st.number_input("GPU TDP (W)", value=700.0, min_value=100.0, step=50.0)
    PUE = st.number_input("PUE", value=1.30, min_value=1.0, max_value=3.0, step=0.05)
    cooling_mode = st.selectbox("Cooling mode", ["air", "liquid"])
    if cooling_mode == "air":
        deltaT = st.number_input("ΔT air (°C)", value=15.0, min_value=1.0, step=1.0)
    else:
        deltaT = st.number_input("ΔT liquid (°C)", value=7.0, min_value=1.0, step=1.0)

    nodes_per_rack = st.number_input("Nodes per rack (0 = skip)", value=0, min_value=0, step=1)

    st.subheader("Network (Step 3)")
    nics_per_node = st.selectbox("NICs per node", [1, 2, 4, 8], index=3,
                                  help="One per GPU for rail-optimised HPC")
    switch_radix = st.selectbox("Switch radix (ports)", [32, 64, 128], index=1,
                                 help="64 = NDR IB / 400GbE standard")
    port_bw_GBs = st.number_input("Port BW (GB/s)", value=50.0, min_value=1.0, step=12.5,
                                   help="400 Gb/s = 50 GB/s")
    oversubscription = st.selectbox("Oversubscription", [1.0, 1.5, 2.0, 4.0], index=0,
                                     help="1.0 = full bisection (HPC default)")
    rail_optimised = st.checkbox("Rail-optimised", value=True,
                                  help="One NIC per GPU, each on its own leaf switch (HPC standard)")

    st.subheader("Storage (Step 4)")
    drive_bw_GBs = st.number_input("Drive seq BW (GB/s)", value=7.0, min_value=0.1, step=0.5,
                                    help="NVMe SSD sequential read/write ≈ 7 GB/s")
    drive_cap_TB = st.number_input("Drive capacity (TB)", value=7.68, min_value=0.1, step=1.0,
                                    help="Enterprise NVMe SSD ≈ 7.68 TB usable")
    drives_per_snode = st.number_input("Drives per storage node", value=24, min_value=1, step=1)
    snode_net_GBs = st.number_input("Storage node net BW (GB/s)", value=25.0, min_value=1.0, step=5.0,
                                     help="NIC bandwidth from storage node to fabric")
    dataset_rep = st.number_input("Dataset replication", value=1, min_value=1, step=1)
    ckpt_rep = st.number_input("Checkpoint replication", value=2, min_value=1, step=1)
    ckpt_keep = st.number_input("Checkpoint generations kept", value=3, min_value=1, step=1)

    st.subheader("Compute Node Spec (Step 5)")
    intra_node_bw_GBs = st.number_input(
        "Intra-node BW (GB/s)", value=900.0, min_value=1.0, step=50.0,
        help="NVLink4 (H100 SXM): 900 GB/s  |  PCIe 5.0 x16: ~128 GB/s"
    )
    gpudirect_rdma = st.checkbox("GPUDirect RDMA", value=True,
                                  help="NIC DMAs directly to/from GPU HBM. Required for line-rate fabric BW.")
    cpu_cores_per_node = st.number_input("CPU cores per node", value=128, min_value=1, step=16,
                                          help="e.g. 2× AMD EPYC 9654 = 192 cores")
    dram_per_node_TB = st.number_input("System DRAM per node (TB)", value=2.0, min_value=0.1, step=0.5,
                                        help="H100 HGX: 2 TB  |  DGX H100: 2 TB")
    root_complex = st.selectbox("Root complex", ["single_socket", "dual_socket", "cascade"], index=1)

    st.subheader("Cost Model (Step 6)")
    st.caption("Ballpark ~2024 prices — adjust to current quotes.")
    gpu_price = st.number_input("GPU unit cost ($)", value=30_000, min_value=0, step=1_000,
                                 help="H100 SXM list ~$30k; street varies widely")
    chassis_price = st.number_input("Node chassis cost ($)", value=15_000, min_value=0, step=1_000,
                                     help="Compute node chassis + CPU(s) + DRAM, excluding GPU and NIC")
    nic_price = st.number_input("NIC unit cost ($)", value=2_500, min_value=0, step=100,
                                 help="Per NIC — NDR IB ~$2-3k, 400GbE ~$1-2k")
    switch_price = st.number_input("Switch unit cost ($)", value=40_000, min_value=0, step=1_000,
                                    help="Per switch (leaf or spine) — NDR 64-port ~$30-50k")
    cable_price = st.number_input("Cable unit cost ($)", value=100, min_value=0, step=10,
                                   help="Per cable — AOC 2m ~$100, DAC cheaper")
    storage_node_price = st.number_input("Storage node cost ($)", value=20_000, min_value=0, step=1_000,
                                          help="Per storage node chassis + CPU + DRAM, excluding drives")
    drive_price = st.number_input("Drive unit cost ($)", value=2_500, min_value=0, step=100,
                                   help="Per NVMe SSD — 7.68TB enterprise ~$2-3k")
    rack_price = st.number_input("Rack unit cost ($)", value=3_000, min_value=0, step=500,
                                  help="Per rack including PDU and cabling infrastructure")
    electricity_rate = st.number_input("Electricity ($/kWh)", value=0.07, min_value=0.0, step=0.01,
                                        help="US industrial average ~$0.07/kWh; DC contracts vary")
    amortization_years = st.number_input("CapEx amortization (years)", value=3.0, min_value=0.5, step=0.5,
                                          help="Typical HPC cluster: 3-5 years")


# ---------------------------------------------------------------------------
# Build typed inputs from sidebar
# ---------------------------------------------------------------------------

workload = Workload(P=P_B * 1e9, Tok=Tok_T * 1e12, T=T_days * 86400.0, c=c)
state = StateBytes(b_w=b_w, b_g=b_g, b_opt=b_opt)
io = IO()
device = Device(F_dev_sust_flop_s=F_dev_TFLOPS * 1e12, B_dev_mem_bytes=B_mem_GB * 1e9)
step = StepWorkingSet(B_step_bytes=B_step_GB * 1e9)
schedule = StepSchedule(Tok_per_step=Tok_per_step_M * 1e6)
update = AlgorithmStepFacts(b_update_per_param=b_g)
fabric = FabricCapability(BW_node_sust_Bps=BW_fabric_GBs * 1e9)
storage = StorageCapability(BW_ckpt_sust_Bps=BW_ckpt_GBs * 1e9)
checkpoint = CheckpointPolicy(seconds_per_ckpt=ckpt_interval_h * 3600.0)
design_inputs = DesignInputs(
    gpus_per_node=int(gpus_per_node),
    eta_compute=eta_compute,
    eta_fabric=eta_fabric,
)
power = PowerInputs(P_gpu_W=P_gpu_W, PUE=PUE)
if cooling_mode == "air":
    thermals = ThermalInputs(mode="air", air=AirCoolingInputs(deltaT_C=deltaT))
else:
    thermals = ThermalInputs(mode="liquid", liquid=LiquidCoolingInputs(deltaT_C=deltaT))
rack = RackInputs(nodes_per_rack=int(nodes_per_rack) if nodes_per_rack > 0 else None)
network_inputs = NetworkInputs(
    nics_per_node=int(nics_per_node),
    switch_radix=int(switch_radix),
    port_bw_Bps=port_bw_GBs * 1e9,
    oversubscription=float(oversubscription),
    rail_optimised=rail_optimised,
)
storage_design_inputs = StorageInputs(
    drive_bw_seq_Bps=drive_bw_GBs * 1e9,
    drive_capacity_bytes=drive_cap_TB * 1e12,
    drives_per_storage_node=int(drives_per_snode),
    storage_net_bw_Bps=snode_net_GBs * 1e9,
    dataset_replication=int(dataset_rep),
    ckpt_replication=int(ckpt_rep),
    ckpt_keep_count=int(ckpt_keep),
)
node_spec = ComputeNodeSpec(
    intra_node_bw_Bps=intra_node_bw_GBs * 1e9,
    gpudirect_rdma=gpudirect_rdma,
    cpu_cores_per_node=int(cpu_cores_per_node),
    dram_bytes_per_node=dram_per_node_TB * 1e12,
    root_complex=root_complex,
)
cost_inputs = CostInputs(
    gpu_unit_cost=float(gpu_price),
    node_chassis_cost=float(chassis_price),
    nic_unit_cost=float(nic_price),
    switch_unit_cost=float(switch_price),
    cable_unit_cost=float(cable_price),
    storage_node_cost=float(storage_node_price),
    drive_unit_cost=float(drive_price),
    rack_unit_cost=float(rack_price),
    electricity_usd_kwh=float(electricity_rate),
    capex_amortization_years=float(amortization_years),
)


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

bundle0 = run_fundamentals(
    workload=workload, state=state, io=io, device=device,
    step=step, schedule=schedule, update=update,
    fabric=fabric, storage=storage, checkpoint=checkpoint,
)

design = run_design(bundle0, G=design_G, inputs=design_inputs)

thermals_result = None
if design["feasible"]:
    thermals_result = run_thermals(
        design, power=power, thermals=thermals, rack=rack,
        T_run_s=workload.T,
    )

network_result = None
if design["feasible"]:
    network_result = run_network(design, network=network_inputs)

storage_result = None
if design["feasible"]:
    storage_result = run_storage(bundle0, design, storage=storage_design_inputs)

bottleneck_result = None
if design["feasible"]:
    bottleneck_result = run_bottleneck(bundle0, design)

bom_result = None
if design["feasible"]:
    bom_result = run_bom(design, thermals_result, network_result, storage_result,
                         bundle0=bundle0, node_spec=node_spec)

cost_result = None
if bom_result is not None:
    cost_result = run_cost(bom_result, thermals_result,
                           bundle0=bundle0, cost=cost_inputs)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _gb(x: float) -> str:
    return f"{x / 1e9:.1f} GB"


def _gbs(x: float) -> str:
    return f"{x / 1e9:.2f} GB/s"


def _tflops(x: float) -> str:
    return f"{x / 1e12:.1f} TFLOP/s"


def _kw(x: float) -> str:
    return f"{x / 1000:.1f} kW"


def _m3s(x: float) -> str:
    return f"{x:.4f} m³/s"


# ---------------------------------------------------------------------------
# Step 0 — Fundamentals
# ---------------------------------------------------------------------------

st.header("Step 0 — Fundamentals")
st.caption("Physics-only invariants; no cluster topology assumed.")

req = bundle0["req"]
mv = bundle0["movement"]
bounds = bundle0["device_bounds"]
ckpt = bundle0["ckpt"]

c0, c1, c2, c3 = st.columns(4)
c0.metric("Compute required", _tflops(req["F_req_flop_s"]))
c1.metric("Token rate", f"{req['R_tok_req_tok_s']:.2e} tok/s")
c2.metric("Model state (1 copy)", _gb(req["B_state_min_bytes"]))
c3.metric("Step time budget", f"{mv['t_step_max_s']:.4f} s")

c0, c1, c2, c3 = st.columns(4)
c0.metric("Dataset BW (planned)", _gbs(req["BW_dataset_plan_Bps"]))
c1.metric("Checkpoint size", _gb(req["S_ckpt_bytes"]))
c2.metric("Update payload/step", _gb(mv["B_update_total_bytes_per_step"]))
c3.metric("Ckpt fraction of run", f"{ckpt['ckpt_fraction_of_run'] * 100:.1f}%")

with st.expander("Device lower bounds"):
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Compute bound", f"{bounds['N_compute_lower_bound']:.0f} devices")
    col_b.metric("State bound", f"{bounds['N_state_memory_lower_bound']:.0f} devices")
    col_c.metric("Instant bound", f"{bounds['N_instant_device_lower_bound']:.0f} devices")
    col_d.metric("N_min", f"{bounds['N_min_lower_bound']:.0f} devices")


# ---------------------------------------------------------------------------
# Step 1 — Design
# ---------------------------------------------------------------------------

st.header("Step 1 — Design")

if not design["feasible"]:
    reason = design.get("no_solution_reason", {})
    st.error(f"**Infeasible:** {reason.get('note', 'No feasible cluster found.')}")
    for hint in reason.get("hints", []):
        st.warning(f"Hint: {hint}")
else:
    sol = design["solution"]
    cl = sol["cluster"]
    par = sol["parallelism"]
    tim = sol["timing"]
    mem = sol["memory"]

    c0, c1, c2 = st.columns(3)
    c0.metric("Total GPUs", cl["G"])
    c1.metric("Nodes", cl["nodes"])
    c2.metric("GPUs / node", cl["gpus_per_node"])

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("DP", par["dp"])
    c1.metric("TP", par["tp"])
    c2.metric("PP", par["pp"])
    c3.metric("Strategy", f"dp={par['dp']} tp={par['tp']} pp={par['pp']}")

    headroom_pct = tim["headroom_s"] / tim["t_step_max_s"] * 100
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Step time", f"{tim['t_step_s']:.4f} s")
    c1.metric("Step budget", f"{tim['t_step_max_s']:.4f} s")
    c2.metric("Headroom", f"{tim['headroom_s']:.4f} s", delta=f"{headroom_pct:.1f}%")
    c3.metric("Compute time", f"{tim['t_compute_s']:.4f} s")

    mem_pct = mem["bytes_per_device"] / mem["B_dev_mem_bytes"] * 100
    c0, c1 = st.columns(2)
    c0.metric("Mem / device", f"{_gb(mem['bytes_per_device'])}  ({mem_pct:.0f}% of cap)")
    c1.metric("Device capacity", _gb(mem["B_dev_mem_bytes"]))

    # Step-time breakdown bar
    t_total = tim["t_step_s"]
    if t_total > 0:
        st.markdown("**Step-time breakdown**")
        compute_pct = tim["t_compute_s"] / t_total * 100
        comm_pct = tim["t_comm_s"] / t_total * 100
        headroom_pct2 = max(0, 100 - compute_pct - comm_pct)
        st.markdown(
            f"`compute {compute_pct:.0f}%` | `comm {comm_pct:.0f}%` | `headroom {headroom_pct2:.0f}%`"
        )


# ---------------------------------------------------------------------------
# Bottleneck Analysis
# ---------------------------------------------------------------------------

st.header("Bottleneck Analysis")

if bottleneck_result is None:
    st.info("Bottleneck analysis skipped (design is infeasible).")
else:
    b   = bottleneck_result["binding"]
    db  = bottleneck_result["device_bounds"]
    st_ = bottleneck_result["step_time"]
    mem = bottleneck_result["memory"]
    comm = bottleneck_result["communication"]

    # Binding constraint banner
    binding_hx = db[b["key"]]["headroom_x"]
    if binding_hx < 1.2:
        st.error(f"**Binding constraint: {b['label'].upper()}** — {binding_hx:.2f}× headroom (very tight)")
    elif binding_hx < 2.0:
        st.warning(f"**Binding constraint: {b['label'].upper()}** — {binding_hx:.2f}× headroom")
    else:
        st.success(f"**Binding constraint: {b['label'].upper()}** — {binding_hx:.2f}× headroom")

    # Device bounds
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Compute bound", f"{db['compute']['N']:.0f} GPUs",
              delta=f"{db['compute']['headroom_x']:.1f}× headroom",
              delta_color="normal" if not db["compute"]["is_binding"] else "off")
    c1.metric("State memory bound", f"{db['state']['N']:.0f} GPUs",
              delta=f"{db['state']['headroom_x']:.1f}× headroom",
              delta_color="normal" if not db["state"]["is_binding"] else "off")
    c2.metric("Instant memory bound", f"{db['instant']['N']:.0f} GPUs",
              delta=f"{db['instant']['headroom_x']:.1f}× headroom",
              delta_color="normal" if not db["instant"]["is_binding"] else "off")
    c3.metric("Actual G / N_min", f"{db['G_over_N_min']:.2f}×",
              help="How much larger the cluster is than the absolute minimum")

    # Step time and memory
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Step time headroom", f"{st_['headroom_pct']:.1f}%",
              delta="tight" if st_["step_time_tight"] else "ok",
              delta_color="inverse" if st_["step_time_tight"] else "normal")
    c1.metric("Compute fraction", f"{st_['compute_pct']:.0f}%")
    c2.metric("Comm fraction", f"{st_['comm_pct']:.0f}%",
              delta="heavy" if comm["comm_heavy"] else "ok",
              delta_color="inverse" if comm["comm_heavy"] else "normal")
    c3.metric("Memory utilisation", f"{mem['utilization_pct']:.0f}%",
              delta="tight" if mem["memory_tight"] else "ok",
              delta_color="inverse" if mem["memory_tight"] else "normal")

    # Recommendations
    recs = bottleneck_result["recommendations"]
    if recs:
        with st.expander(f"Recommendations ({len(recs)})", expanded=True):
            for r in recs:
                st.markdown(f"→ {r}")


# ---------------------------------------------------------------------------
# Step 2 — Power & Thermals
# ---------------------------------------------------------------------------

st.header("Step 2 — Power & Thermals")

if thermals_result is None:
    st.info("Step 2 skipped (design is infeasible).")
else:
    pwr = thermals_result["power"]
    heat = thermals_result["heat"]
    cool = thermals_result["cooling"]

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("IT Power", _kw(pwr["P_IT_W"]))
    c1.metric("Facility Power", _kw(pwr["P_facility_W"]), delta=f"PUE {pwr['PUE']:.2f}")
    c2.metric("Heat load", _kw(heat["Qdot_W"]))
    if cool["mode"] == "air":
        c3.metric("Airflow", _m3s(cool["vol_flow_m3_s"]))
    else:
        c3.metric("Coolant flow", _m3s(cool["vol_flow_m3_s"]))

    rk = thermals_result.get("rack")
    if rk:
        st.markdown(f"**Racks:** {rk['racks']}  —  {_kw(rk['P_IT_per_rack_W'])}/rack")
        if "rack_power_ok" in rk:
            if rk["rack_power_ok"]:
                st.success(f"Rack power OK (margin {_kw(rk['rack_power_margin_W'])})")
            else:
                st.error(f"Rack power EXCEEDS limit by {_kw(-rk['rack_power_margin_W'])}")

    en = thermals_result.get("energy")
    if en:
        c0, c1 = st.columns(2)
        c0.metric("Energy (IT)", f"{en['E_IT_kWh']:,.0f} kWh")
        c1.metric("Energy (facility)", f"{en['E_facility_kWh']:,.0f} kWh")


# ---------------------------------------------------------------------------
# Step 3 — Network
# ---------------------------------------------------------------------------

st.header("Step 3 — Network")

if network_result is None:
    st.info("Step 3 skipped (design is infeasible).")
else:
    sw = network_result["switches"]
    bw = network_result["bandwidth"]
    ca = network_result["cables"]
    cons = network_result["consistency"]

    topology_label = "Rail-optimised leaf/spine" if network_result["topology"] == "rail_optimised_leaf_spine" else "Leaf/spine"

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Topology", topology_label)
    c1.metric("Leaf switches", sw["num_leaves"],
              help=f"{sw['leaf_downlinks']} server ports / {sw['leaf_uplinks']} uplink ports each")
    c2.metric("Spine switches", sw["num_spines"])
    c3.metric("Total switches", sw["total"])

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Per-node BW (effective)", _gbs(bw["bw_per_node_effective_Bps"]))
    c1.metric("Per-node BW (raw)", _gbs(bw["bw_per_node_raw_Bps"]))
    c2.metric("Bisection BW", f"{bw['bisection_bw_Tbps']:.2f} Tb/s")
    c3.metric("Total cables", ca["total"],
              help=f"{ca['server_to_leaf']} server→leaf  +  {ca['leaf_to_spine']} leaf→spine")

    if not network_result["two_tier_feasible"]:
        st.warning(
            f"Node count exceeds 2-tier capacity ({network_result['max_nodes_2tier']} nodes "
            f"with radix-{sw['leaf_downlinks'] + sw['leaf_uplinks']}). A 3-tier fat-tree is needed."
        )

    if not cons["ok"]:
        st.warning(cons["note"])
    else:
        ratio = cons["ratio"]
        st.success(
            f"Fabric BW consistent with Step 1 assumption  "
            f"(derived {_gbs(cons['step3_derived_bw_node_Bps'])} vs assumed {_gbs(cons['step1_assumed_bw_node_Bps'])}, "
            f"ratio {ratio:.2f}×)"
        )

    with st.expander("Cable schedule"):
        st.markdown(f"- **Server → leaf:** {ca['server_to_leaf']} cables")
        st.markdown(f"- **Leaf → spine:** {ca['leaf_to_spine']} cables")
        st.markdown(f"- **Total:** {ca['total']} cables")


# ---------------------------------------------------------------------------
# Step 4 — Storage
# ---------------------------------------------------------------------------

st.header("Step 4 — Storage")

if storage_result is None:
    st.info("Step 4 skipped (design is infeasible).")
else:
    nd = storage_result["nodes"]
    dp = storage_result["dataset_pool"]
    cp = storage_result["checkpoint_pool"]
    nw = storage_result["network"]
    cons = storage_result["consistency"]

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Storage nodes", nd["num_storage_nodes"])
    c1.metric("Total drives", nd["total_drives"],
              help=f"{nd['drives_per_node']} drives/node")
    c2.metric("Dataset drives", dp["drives"],
              help=f"BW-bound: {dp['drives_for_bw']}  Cap-bound: {dp['drives_for_capacity']}")
    c3.metric("Checkpoint drives", cp["drives"],
              help=f"BW-bound: {cp['drives_for_bw']}  Cap-bound: {cp['drives_for_capacity']}")

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Dataset pool BW", _gbs(dp["aggregate_bw_Bps"]))
    c1.metric("Checkpoint pool BW", _gbs(cp["aggregate_bw_Bps"]))
    c2.metric("Storage net BW", _gbs(nw["aggregate_net_bw_Bps"]),
              delta=f"{nw['net_ratio']:.2f}× required")
    c3.metric("Dataset stored", _gb(dp["bytes_stored"]),
              help=f"×{dp['replication']} replication")

    if not nw["net_ok"]:
        st.warning(
            f"Storage network BW insufficient: have {_gbs(nw['aggregate_net_bw_Bps'])}, "
            f"need {_gbs(nw['required_net_bw_Bps'])}. Add more storage nodes or higher-BW NICs."
        )

    if not cons["ok"]:
        st.warning(cons["note"])
    else:
        st.success(
            f"Checkpoint pool BW consistent with Step 0 assumption  "
            f"({_gbs(cons['ckpt_pool_bw_Bps'])} derived vs {_gbs(cons['step0_assumed_ckpt_bw_Bps'])} assumed, "
            f"ratio {cons['ratio']:.2f}×)"
        )

    with st.expander("Storage schedule"):
        st.markdown(f"**Dataset pool:** {dp['drives']} drives × {_gbs(storage_design_inputs.drive_bw_seq_Bps)} = {_gbs(dp['aggregate_bw_Bps'])} aggregate")
        st.markdown(f"  - Bytes stored: {_gb(dp['bytes_stored'])} (×{dp['replication']} replication)")
        st.markdown(f"**Checkpoint pool:** {cp['drives']} drives × {_gbs(storage_design_inputs.drive_bw_seq_Bps)} = {_gbs(cp['aggregate_bw_Bps'])} aggregate")
        st.markdown(f"  - Bytes stored: {_gb(cp['bytes_stored'])} (×{cp['replication']} replication, {cp['keep_count']} generations)")
        st.markdown(f"**Storage nodes:** {nd['num_storage_nodes']} × {int(drives_per_snode)} drives/node = {nd['total_drives']} drives")


# ---------------------------------------------------------------------------
# Step 5 — Bill of Materials
# ---------------------------------------------------------------------------

st.header("Step 5 — Bill of Materials")

if bom_result is None:
    st.info("Step 5 skipped (design is infeasible).")
else:
    t = bom_result["totals"]
    c = bom_result["compute"]
    nw = bom_result["network"]
    sb = bom_result["storage"]
    fac = bom_result["facility"]
    ns = bom_result["node_spec"]
    checks = bom_result["checks"]

    # Consistency warnings — prominent at the top
    for w in bom_result["warnings"]:
        st.warning(w)

    # Node spec summary row
    gdrdma_label = "✓ GPUDirect RDMA" if ns["gpudirect_rdma"] else "✗ No GPUDirect RDMA"
    st.caption(
        f"Node: {ns['root_complex']}  |  "
        f"NVLink {ns['intra_node_bw_Bps']/1e9:.0f} GB/s/node "
        f"({ns['intra_node_bw_per_gpu_Bps']/1e9:.0f} GB/s/GPU)  |  "
        f"{gdrdma_label}  |  "
        f"{ns['cpu_cores_per_node']} cores/node  |  "
        f"{ns['dram_bytes_per_node']/1e12:.1f} TB DRAM/node"
    )

    # Headline metrics
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("All nodes", t["all_nodes"],
              help=f"{t['compute_nodes']} compute + {t['storage_nodes']} storage")
    c1.metric("GPUs", t["gpus"],
              help=f"{c['gpus_per_node']} per node")
    c2.metric("Total switches", t["switches"] if t["switches"] is not None else "—")
    c3.metric("Total cables", t["cables"] if t["cables"] is not None else "—")

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("NICs", t["nics"] if t["nics"] is not None else "—",
              help=f"{nw['nics_per_node']} per node" if nw else None)
    c1.metric("Total drives", t["drives"] if t["drives"] is not None else "—")
    c2.metric("Storage nodes", t["storage_nodes"])
    c3.metric("Racks", t["racks"] if t["racks"] is not None else "—")

    c0, c1, c2 = st.columns(3)
    c0.metric("CPU cores (total)", t["cpu_cores"],
              help=f"{ns['cpu_cores_per_node']} per node")
    c1.metric("System DRAM (total)", f"{t['dram_bytes']/1e12:.1f} TB",
              help=f"{ns['dram_bytes_per_node']/1e12:.1f} TB per node")
    c2.metric("NVLink / intra-node",
              f"{ns['intra_node_bw_Bps']/1e9:.0f} GB/s node",
              delta="OK" if checks["nvlink_ok"] else "BOTTLENECK",
              delta_color="normal" if checks["nvlink_ok"] else "inverse")

    # Detailed table
    with st.expander("Full component list"):
        rows = []
        rows.append(("**Compute nodes**", t["compute_nodes"], ""))
        rows.append((f"  └─ GPUs", t["gpus"], f"{c['gpus_per_node']} per node"))
        if nw is not None:
            rows.append((f"  └─ NICs", nw["nics"], f"{nw['nics_per_node']} per node"))
            rows.append(("**Leaf switches**", nw["leaf_switches"], ""))
            rows.append(("**Spine switches**", nw["spine_switches"], ""))
            rows.append(("**Total switches**", nw["total_switches"], ""))
            rows.append(("Cables (server→leaf)", nw["cables_server_to_leaf"], ""))
            rows.append(("Cables (leaf→spine)", nw["cables_leaf_to_spine"], ""))
            rows.append(("**Total cables**", nw["total_cables"], ""))
        if sb is not None:
            rows.append(("**Storage nodes**", sb["storage_nodes"],
                         f"{sb['drives_per_storage_node']} drives/node"))
            rows.append(("  └─ Dataset drives", sb["dataset_drives"], ""))
            rows.append(("  └─ Checkpoint drives", sb["checkpoint_drives"], ""))
            rows.append(("  └─ Total drives", sb["total_drives"], ""))
        if fac is not None and fac["racks"] is not None:
            rows.append(("**Racks**", fac["racks"],
                         f"{fac['nodes_per_rack']} nodes/rack" if fac["nodes_per_rack"] else ""))
        rows.append(("**ALL NODES (compute+storage)**", t["all_nodes"], ""))

        col_w = 3
        for label, qty, note in rows:
            sep = "—" * 38 if label.startswith("**ALL") else None
            if sep:
                st.markdown(f"---")
            cols = st.columns([col_w, 1, 2])
            cols[0].markdown(label)
            cols[1].markdown(f"**{qty}**")
            if note:
                cols[2].caption(note)


# ---------------------------------------------------------------------------
# Step 6 — Cost Model
# ---------------------------------------------------------------------------

st.header("Step 6 — Cost Model")

if cost_result is None:
    st.info("Step 6 skipped (design is infeasible).")
else:
    cx = cost_result["capex"]
    op = cost_result["opex"]
    rc = cost_result["run_cost"]
    ct = cost_result["cost_per_token"]
    assum = cost_result["assumptions"]

    def _usd_fmt(x: float) -> str:
        if x >= 1e9:
            return f"${x/1e9:.2f}B"
        if x >= 1e6:
            return f"${x/1e6:.2f}M"
        if x >= 1e3:
            return f"${x/1e3:.1f}k"
        return f"${x:.2f}"

    # Headline numbers
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Total CapEx", _usd_fmt(cx["total"]))
    c1.metric("Energy cost (run)", _usd_fmt(op["energy_cost"]) if op["energy_cost"] else "—")
    c2.metric("Run cost (amortised)", _usd_fmt(rc["total_amortised"]) if rc["total_amortised"] else "—",
              help=f"CapEx amortised over {assum['amortization_years']:.0f} years + energy")
    c3.metric("Run cost (full CapEx)", _usd_fmt(rc["total_full_capex"]) if rc["total_full_capex"] else "—",
              help="Full hardware cost allocated to this run + energy")

    if ct["amortised"] is not None:
        c0, c1, c2 = st.columns(3)
        c0.metric("Cost / token (amortised)", f"${ct['amortised']:.6f}")
        c1.metric("Cost / token (full CapEx)", f"${ct['full_capex']:.6f}")
        c2.metric("Tokens", f"{ct['Tok']:.2e}")

    # CapEx breakdown
    with st.expander("CapEx breakdown"):
        items = [
            ("GPUs", cx["gpus"]),
            ("Node chassis", cx["node_chassis"]),
            ("NICs", cx["nics"]),
            ("Switches", cx["switches"]),
            ("Cables", cx["cables"]),
            ("Storage nodes", cx["storage_nodes"]),
            ("Drives", cx["drives"]),
            ("Racks", cx["racks"]),
        ]
        for label, val in items:
            if val > 0:
                pct = val / cx["total"] * 100
                cols = st.columns([3, 1, 1])
                cols[0].markdown(label)
                cols[1].markdown(f"**{_usd_fmt(val)}**")
                cols[2].caption(f"{pct:.1f}%")
        st.markdown("---")
        cols = st.columns([3, 1, 1])
        cols[0].markdown("**Total CapEx**")
        cols[1].markdown(f"**{_usd_fmt(cx['total'])}**")

    if op.get("note"):
        st.info(op["note"])
    if ct.get("note"):
        st.info(ct["note"])


# ---------------------------------------------------------------------------
# Debug expander
# ---------------------------------------------------------------------------

with st.expander("Raw bundles (debug)"):
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Step 0", "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"])
    with tab0:
        st.json(bundle0)
    with tab1:
        st.json(design)
    with tab2:
        st.json(thermals_result or {})
    with tab3:
        st.json(network_result or {})
    with tab4:
        st.json(storage_result or {})
    with tab5:
        st.json(bom_result or {})
    with tab6:
        st.json(cost_result or {})
