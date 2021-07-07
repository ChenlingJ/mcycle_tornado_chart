import mcycle as mc
from math import nan
import CoolProp as CP
import numpy as np


def my_heat_exchanger(
    flowConfig=mc.HxFlowConfig(
        sense=mc.COUNTERFLOW, passes=1, verticalWf=True, verticalSf=True
    ),
    flowInWf=None,
    flowInSf=None,
    flowOutWf=None,
    flowOutSf=None,
    ambient=None,
    NPlate=3,
    sizeAttr="NPlate",
    sizeBounds=[3, 100],
    sizeUnitsBounds=[1e-5, 10.0],
    name="HxPlate instance",
    **kwargs,
):
    hx = mc.HxPlate(
        flowConfig=flowConfig,
        NPlate=NPlate,
        RfWf=0,
        RfSf=0,
        plate=mc.stainlessSteel_316(),
        tPlate=0.424e-3,
        geomWf=mc.GeomHxPlateChevron(1.096e-3, 60, 10e-3, 1.117),
        geomSf=mc.GeomHxPlateChevron(1.096e-3, 60, 10e-3, 1.117),
        L=269e-3,
        W=95e-3,
        portWf=mc.Port(d=0.0125),
        portSf=mc.Port(d=0.0125),
        coeffs_LPlate=[0.056, 1],
        coeffs_WPlate=[0, 1],
        coeffs_mass=[
            1.0 / (0.325 * 0.095 * 0.424e-3),
            0.09 / (0.325 * 0.095 * 0.424e-3),
        ],
        efficiencyThermal=1.0,
        flowInWf=flowInWf,
        flowInSf=flowInSf,
        flowOutWf=flowOutWf,
        flowOutSf=flowOutSf,
        ambient=ambient,
        sizeAttr=sizeAttr,
        sizeBounds=sizeBounds,
        sizeUnitsBounds=sizeUnitsBounds,
        name=name,
    )
    hx.update(kwargs)
    return hx


def do_plot(
    actually_plot=False,
    updates=None,
    update_fn=None,
    plot_name="my_plot",
    new_figure=True,
):
    if updates is None:
        updates = {}

    # ************************************************
    # Set MCycle defaults
    # ************************************************
    mc.defaults.PLOT_DIR = ""
    mc.defaults.PLOT_DPI = 200
    mc.defaults.check()

    wf = mc.FlowState(
        fluid="n-Pentane", m=35, inputPair=mc.PT_INPUTS, input1=mc.atm2Pa(1), input2=298
    )
    sf_mass_flow_rate = 50
    sf_fluid = "Water"
    sf_pressure = 1e6
    sf_hot = mc.FlowState(
        fluid=sf_fluid,
        m=sf_mass_flow_rate,
        inputPair=mc.PT_INPUTS,
        input1=sf_pressure,
        input2=mc.degC2K(120),
    )
    sf_cool = mc.FlowState(
        fluid=sf_fluid,
        m=sf_mass_flow_rate,
        inputPair=mc.PT_INPUTS,
        input1=sf_pressure,
        input2=mc.degC2K(60),
    )
    exp = mc.ExpBasic(pRatio=1, efficiencyIsentropic=0.9, sizeAttr="pRatio")
    cond = mc.ClrBasic(
        constraint=mc.CONSTANT_P, QCool=1, efficiencyThermal=1.0, sizeAttr="Q"
    )
    comp = mc.CompBasic(pRatio=1, efficiencyIsentropic=0.85, sizeAttr="pRatio")
    evap = my_heat_exchanger(
        sizeAttr="flowOutSf",
        NPlate=30,
        flowConfig=mc.HxFlowConfig(
            sense=mc.COUNTERFLOW, passes=1, verticalWf=True, verticalSf=True
        ),
        flowInSf=sf_hot,
        ambient=mc.FlowState(
            fluid="Air",
            inputPair=mc.PT_INPUTS,
            input1=mc.atm2Pa(1),
            input2=mc.degC2K(20),
        ),
    )
    config = mc.Config()
    cycle = mc.RankineBasic(
        wf=wf, evap=evap, exp=exp, cond=cond, comp=comp, config=config
    )
    TCond = mc.degC2K(25)
    pEvap = mc.bar2Pa(14.56)
    cycle.update(
        {
            "pEvap": pEvap,
            "superheat": 10.0,
            "TCond": TCond,
            "subcool": 2.0,
        }
    )

    if update_fn is None:
        cycle.update(updates)
    else:
        for value in updates.values():
            update_fn(cycle, value)

    @mc.timer
    def plot_cycle():
        cycle.sizeSetup(unitiseEvap=False, unitiseCond=False)
        cycle.plot(
            graph="Ts",  # either 'Ts' or 'ph'
            title="T-s Diagram for Organic Rankine Cycle",  # graph title
            satCurve=True,  # display saturation curve
            newFig=new_figure,  # create a new figure
            show=False,  # show the figure
            savefig=True,  # save the figure
            savefig_name="foobar",
        )

    if actually_plot:
        plot_cycle()

    cycle.size()

    return cycle


cycle = do_plot(actually_plot=True)
base = cycle.efficiencyGlobal()
lo_hi = {}


def make_plots(param_ranges):
    for prop_unit in param_ranges:
        prop, label, unit = prop_unit.split("|")
        print(f"Making plot for {prop}...")
        prop_values = param_ranges[prop_unit]
        if isinstance(prop_values, tuple) and isinstance(prop_values[0], float):
            factor, prop_values = prop_values
        else:
            factor = 1.0
        update_fn = None
        if isinstance(prop_values, tuple) and callable(prop_values[1]):
            prop_values, update_fn = prop_values
        prop_values /= factor
        for i in range(len(prop_values)):
            prop_values[i] = round(prop_values[i], 2)
        efficiency_values = []
        for value in prop_values:
            cycle = do_plot(
                updates={
                    prop: value * factor,
                },
                update_fn=update_fn,
            )
            efficiency_values.append(round(cycle.efficiencyGlobal(), 2))

        from matplotlib import pyplot as plt

        plt.figure()
        plt.plot(prop_values, efficiency_values)
        plt.xlabel(f"{label} ({unit})")
        plt.ylabel("Global efficiency")
        plt.title(f"Varying {prop}")
        plt.savefig(f"varying_{prop}.png")
        min_pv = min(
            prop_values, key=lambda v: efficiency_values[list(prop_values).index(v)]
        )
        max_pv = max(
            prop_values, key=lambda v: efficiency_values[list(prop_values).index(v)]
        )
        lo_hi[f"{label} ({min_pv}-{max_pv} {unit})"] = (
            min(efficiency_values),
            max(efficiency_values),
        )
        plt.close()


make_plots(
    {
        "wf.m|Working fluid mass flow rate|kg/s": np.linspace(25, 43, 10),
        "evap.flowInSf.m|Secondary fluid mass flow rate|kg/s": np.linspace(41, 60, 10),
        "superheat|Superheat delta temperature|K": np.linspace(5, 62, 10),
        "subcool|Subcool delta temperature|K": np.linspace(0, 12, 10),
        "TCond|Condenser temperature|K": np.linspace(mc.degC2K(0), mc.degC2K(60), 10),
        "pEvap|Evaporator (HX) pressure|MPa": (
            1e6,
            np.linspace(mc.bar2Pa(10), mc.bar2Pa(31), 10),
        ),
        #     "evap.NPlate|Number of HX plates": np.array(list(range(20,40)), dtype=float),
        "evap.flowInSf.T|Secondary fluid temperature into HX|K": (
            np.linspace(mc.degC2K(110), mc.degC2K(130), 10),
            lambda cycle, value: setattr(
                cycle.evap,
                "flowInSf",
                cycle.evap.flowInSf.copyUpdateState(
                    mc.PT_INPUTS, cycle.evap.flowInSf.p(), value
                ),
            ),
        ),
        "evap.flowInSf.p|Secondary fluid pressure|MPa": (
            1e6,
            (
                np.linspace(2e5, 1e7, 10),
                lambda cycle, value: setattr(
                    cycle.evap,
                    "flowInSf",
                    cycle.evap.flowInSf.copyUpdateState(
                        mc.PT_INPUTS, value, cycle.evap.flowInSf.T()
                    ),
                ),
            ),
        ),
    }
)


import numpy as np
from matplotlib import pyplot as plt

plt.figure()

###############################################################################
# The data (change all of this to your actual data, this is just a mockup)
variables = list(sorted(lo_hi.keys(), key=lambda key: -(lo_hi[key][1] - lo_hi[key][0])))

lows = np.array(list(map(lambda key: lo_hi[key][0], variables)))

values = np.array(list(map(lambda key: lo_hi[key][1], variables)))

min_low = min(lows)
max_hi = max(values)

###############################################################################
# The actual drawing part

# The y position for each variable
ys = range(len(values))[::-1]  # top to bottom

# Plot the bars, one by one
for y, low, value in zip(ys, lows, values):
    # The width of the 'low' and 'high' pieces
    low_width = base - low
    high_width = low + value - base

    # Each bar is a "broken" horizontal bar chart
    plt.broken_barh(
        [(low, low_width), (base, high_width)],
        (y - 0.4, 0.8),
        facecolors=["white", "white"],  # Try different colors if you like
        edgecolors=["black", "black"],
        linewidth=1,
    )

    # Display the value as text. It should be positioned in the center of
    # the 'high' bar, except if there isn't any room there, then it should be
    # next to bar instead.
    x = base + high_width / 2
    if x <= base + 50:
        x = base + high_width + 50
    plt.text(x, y, str(value), va="center", ha="center")

# Draw a vertical line down the middle
plt.axvline(base, color="black")

# Position the x-axis on the top, hide all the other spines (=axis lines)
axes = plt.gca()  # (gca = get current axes)
axes.spines["left"].set_visible(False)
axes.spines["right"].set_visible(False)
axes.spines["bottom"].set_visible(False)
axes.xaxis.set_ticks_position("top")
axes.xaxis.set_label_position("top")

plt.xlabel("Global efficiency")
plt.gcf().subplots_adjust(left=0.45)

# Make the y-axis display the variables
plt.yticks(ys, variables)

# Set the portion of the x- and y-axes to show
plt.xlim(0, max_hi + 0.2)
plt.ylim(-1, len(variables))


plt.show()


for state in [
    "state1",
    "state20",
    "state21",
    "state3",
    "state4",
    "state51",
    "state50",
    "state6",
]:
    print(state, getattr(cycle, state).p())
