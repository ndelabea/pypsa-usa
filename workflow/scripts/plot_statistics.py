"""
Plots static and interactive charts to analyze system results.

**Inputs**

A solved network

**Outputs**

System level charts for:
    - Hourly production
    - Generator costs
    - Generator capacity

    .. image:: _static/plots/production-area.png
        :scale: 33 %

    .. image:: _static/plots/costs-bar.png
        :scale: 33 %

    .. image:: _static/plots/capacity-bar.png
        :scale: 33 %

Emission charts for:
    - Accumulated emissions

    .. image:: _static/plots/emissions-area.png
        :scale: 33 %
"""

import logging
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from _helpers import configure_logging
from add_electricity import sanitize_carriers
from plot_network_maps import get_color_palette
from summary import (
    get_demand_timeseries,
    get_energy_timeseries,
    get_fuel_costs,
    get_generator_marginal_costs,
    get_node_emissions_timeseries,
    get_tech_emissions_timeseries,
)

logger = logging.getLogger(__name__)

# Global Plotting Settings
TITLE_SIZE = 16


def create_title(title: str, **wildcards) -> str:
    """
    Standardizes wildcard writing in titles.

    Arguments:
        title: str
            Title of chart to plot
        **wildcards
            any wildcards to add to title
    """
    w = []
    for wildcard, value in wildcards.items():
        if wildcard == "interconnect":
            w.append(f"interconnect = {value}")
        elif wildcard == "clusters":
            w.append(f"#clusters = {value}")
        elif wildcard == "ll":
            w.append(f"ll = {value}")
        elif wildcard == "opts":
            w.append(f"opts = {value}")
        elif wildcard == "sector":
            w.append(f"sectors = {value}")
    wildcards_joined = " | ".join(w)
    return f"{title} \n ({wildcards_joined})"


def stacked_bar_horizons(
    stats,
    variable,
    variable_units,
    carriers,
):
    carriers = carriers.set_index("nice_name")
    colors_ = carriers["color"]
    carriers_legend = carriers  # to track which carriers have non-zero values
    # Create subplots
    planning_horizons = stats[next(iter(stats.keys()))].columns
    fig, axes = plt.subplots(
        nrows=len(planning_horizons),
        ncols=1,
        figsize=(8, 1.2 * len(planning_horizons)),
        sharex=True,
    )

    # Ensure axes is always iterable (even if there's only one planning horizon)
    if len(planning_horizons) == 1:
        axes = [axes]

    # Loop through each planning horizon
    for ax, horizon in zip(axes, planning_horizons):
        y_positions = np.arange(len(stats))  # One position for each scenario
        for j, (scenario, df) in enumerate(stats.items()):
            bottoms = np.zeros(
                len(df.columns),
            )  # Initialize the bottom positions for stacking
            # Stack the technologies for each scenario
            for i, technology in enumerate(df.index.unique()):
                values = df.loc[technology, horizon]
                values = values / (1e3) if "GW" in variable_units else values
                ax.barh(
                    y_positions[j],
                    values,
                    left=bottoms[j],
                    color=colors_[technology],
                    label=technology if j == 0 else "",
                )
                bottoms[j] += values
                carriers_legend.loc[technology, "value"] = values

        # Set the title for each subplot
        ax.text(
            1.01,
            0.5,
            f"{horizon}",
            transform=ax.transAxes,
            va="center",
            rotation="vertical",
        )
        ax.set_yticks(y_positions)  # Positioning scenarios on the y-axis
        ax.set_yticklabels(stats.keys())  # Labeling y-axis with scenario names
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Create legend handles and labels from the carriers DataFrame
    carriers_legend = carriers_legend[carriers_legend["value"] > 0.01]
    colors_ = carriers_legend["color"]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors_[tech]) for tech in carriers_legend.index]
    # fig.legend(handles=legend_handles, labels=carriers.index.tolist(), loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=4, title='Technologies')
    ax.legend(
        handles=legend_handles,
        labels=carriers_legend.index.tolist(),
        loc="upper center",
        bbox_to_anchor=(0.5, -1.3),
        ncol=4,
        title="Technologies",
    )

    fig.subplots_adjust(hspace=0, bottom=0.5)
    fig.suptitle(f"{variable}", fontsize=12, fontweight="bold")
    plt.xlabel(f"{variable} {variable_units}")
    # fig.tight_layout()
    # plt.show(block=True)
    return fig


#### Bar Plots ####
def plot_capacity_additions_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plots base capacity vs optimal capacity as a bar chart."""
    existing_capacity = n.generators.groupby("carrier").p_nom.sum().round(0)
    existing_capacity = existing_capacity.to_frame(name="Existing Capacity")
    storage_units = n.storage_units.groupby("carrier").p_nom.sum().round(0)
    storage_units = storage_units.to_frame(name="Existing Capacity")
    links_filtered = n.links[n.links.carrier.isin(["H2 fuel cell", "H2 electrolysis"])]
    links = links_filtered.groupby("carrier").p_nom.sum().round(0)
    links = links.to_frame(name="Existing Capacity")
    existing_capacity = pd.concat([existing_capacity, storage_units, links])
    existing_capacity.index = existing_capacity.index.map(n.carriers.nice_name)

    optimal_capacity = n.statistics.optimal_capacity()
    optimal_capacity = optimal_capacity[optimal_capacity.index.get_level_values(0).isin(["Generator", "StorageUnit", "Link"])]
    optimal_capacity.index = optimal_capacity.index.droplevel(0)
    optimal_capacity = optimal_capacity.reset_index()
    optimal_capacity = optimal_capacity.rename(columns={"index": "carrier"})
    is_link = optimal_capacity["carrier"].isin(n.links.carrier.unique())
    is_h2_link = optimal_capacity["carrier"].isin(["H2 fuel cell", "H2 electrolysis"])
    optimal_capacity = optimal_capacity[~is_link | is_h2_link]
    optimal_capacity = optimal_capacity[~optimal_capacity["carrier"].isin(["Ac", "Dc"])]

    optimal_capacity = optimal_capacity.set_index("carrier")
    optimal_capacity.insert(0, "Existing", existing_capacity["Existing Capacity"])
    optimal_capacity = optimal_capacity.fillna(0)

    stats = {"": optimal_capacity}
    variable = "Optimal Capacity"
    variable_units = " GW"
    fig_ = stacked_bar_horizons(stats, variable, variable_units, n.carriers)
    fig_.savefig(save)
    plt.close()

def plot_energy_capacity_additions_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plots base capacity vs optimal energy as a bar chart."""
    if n.stores.empty:
        existing_energy_capacity = pd.DataFrame(columns=["Existing Energy Capacity"])
    else:
        existing_energy_capacity = n.stores.groupby("carrier").e_nom.sum().round(0)
        existing_energy_capacity = existing_energy_capacity.to_frame(name="Existing Energy Capacity")

    existing_storage_units = n.storage_units.groupby("carrier").p_nom.sum().round(0)
    existing_storage_units = existing_storage_units.to_frame(name="Existing Capacity")
    existing_energy_capacity = pd.concat([existing_energy_capacity, existing_storage_units])
    existing_energy_capacity.index = existing_energy_capacity.index.map(n.carriers.nice_name)

    optimal_energy_capacity = n.statistics.optimal_capacity()
    optimal_energy_capacity = optimal_energy_capacity[
        optimal_energy_capacity.index.get_level_values(0).isin(["StorageUnit", "Store"])
    ]
    optimal_energy_capacity.index = optimal_energy_capacity.index.droplevel(0)
    optimal_energy_capacity = optimal_energy_capacity.reset_index()
    optimal_energy_capacity = optimal_energy_capacity.rename(columns={"index": "carrier"})

    optimal_energy_capacity = optimal_energy_capacity.set_index("carrier")
    optimal_energy_capacity.insert(
        0,
        "Existing",
        existing_energy_capacity.get("Existing Energy Capacity", pd.Series(dtype=float))
    )
    optimal_energy_capacity = optimal_energy_capacity.fillna(0)

    stats = {"": optimal_energy_capacity}
    variable = "Optimal Energy Capacity"
    variable_units = " GWh"
    fig_ = stacked_bar_horizons(stats, variable, variable_units, n.carriers)
    fig_.savefig(save)
    plt.close()


def plot_production_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """Plot diaptch per carrier."""
    energy_mix = n.statistics.supply().round(0)
    energy_mix = energy_mix[
        energy_mix.index.get_level_values("component").isin(
            ["Generator", "StorageUnit", "Store"],
        )
    ]
    energy_mix.index = energy_mix.index.droplevel(0)
    energy_mix = energy_mix.fillna(0)
    stats = {"": energy_mix}
    variable = "Energy Mix"
    variable_units = " GWh"

    fig_ = stacked_bar_horizons(stats, variable, variable_units, n.carriers)
    fig_.savefig(save)
    plt.close()


def plot_global_constraint_shadow_prices(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """Plots shadow prices on global constraints."""
    shadow_prices = n.global_constraints.mu.round(3).reset_index()
    print(shadow_prices.mu) ################################################################
    # plot data
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.barplot(
        y=shadow_prices.GlobalConstraint,
        x=shadow_prices.mu,
        data=shadow_prices,
        color="purple",
        ax=ax,
    )

    ax.set_title(create_title("Shadow Prices on Constraints", **wildcards))
    ax.set_ylabel("")
    ax.set_xlabel("Shadow Price [$/MWh]")
    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def get_currently_installed_capacity(n: pypsa.Network) -> pd.DataFrame:
    """Returns a DataFrame with the currently installed capacity for each carrier and nerc region."""
    n.generators["nerc_reg"] = n.generators.bus.map(n.buses.nerc_reg)
    existing_capacity = n.generators.groupby(["nerc_reg", "carrier"]).p_nom.sum().round(0)
    existing_capacity = existing_capacity.to_frame(name="Existing")
    n.storage_units["nerc_reg"] = n.storage_units.bus.map(n.buses.nerc_reg)
    storage_units = n.storage_units.groupby(["nerc_reg", "carrier"]).p_nom.sum().round(0)
    storage_units = storage_units.to_frame(name="Existing")

    for bus in n.buses.index:
        if " H2" in bus:
            base_bus = bus.replace(" H2", "")
            if base_bus in n.buses.index:
                n.buses.loc[bus, "nerc_reg"] = n.buses.loc[base_bus, "nerc_reg"]
    n.links["nerc_reg"] = n.links.apply(lambda x: 
        n.buses.nerc_reg.get(x.bus0, np.nan) if x.carrier == "H2 fuel cell" 
        else n.buses.nerc_reg.get(x.bus1, np.nan) if x.carrier == "H2 electrolysis" 
        else np.nan, 
        axis=1
    )
    
    links_filtered = n.links[n.links.carrier.isin(["H2 fuel cell", "H2 electrolysis"])]
    links = links_filtered.groupby(["nerc_reg", "carrier"]).p_nom.sum().round(0)
    links = links.to_frame(name="Existing")
    existing_capacity = pd.concat([existing_capacity, storage_units, links])

    # Groupby regions and carriers, then fix indexing
    existing_capacity = existing_capacity.groupby(existing_capacity.index).sum()
    existing_capacity = existing_capacity.reset_index()
    existing_capacity[["Region", "Carrier"]] = pd.DataFrame(
        existing_capacity["index"].tolist(),
        index=existing_capacity.index,
    )
    existing_capacity = existing_capacity.drop(columns="index")
    existing_capacity = existing_capacity.set_index(["Region", "Carrier"])

    nn_carriers = existing_capacity.index.get_level_values(1).map(n.carriers.nice_name)
    existing_capacity = existing_capacity.droplevel(1)
    existing_capacity = existing_capacity.set_index(nn_carriers, append=True)

    return existing_capacity


def get_currently_installed_energy_capacity(n: pypsa.Network) -> pd.DataFrame:
    """Returns a DataFrame with the currently installed energy capacity for each carrier and nerc region."""
    # Handle empty stores gracefully
    if n.stores.empty:
        existing_energy_capacity = pd.DataFrame(columns=["Existing"])
    else:
        n.stores["nerc_reg"] = n.stores.bus.map(n.buses.nerc_reg)
        existing_energy_capacity = (
            n.stores.groupby(["nerc_reg", "carrier"]).e_nom.sum().round(0)
        )
        existing_energy_capacity = existing_energy_capacity.to_frame(name="Existing")

    n.storage_units["nerc_reg"] = n.storage_units.bus.map(n.buses.nerc_reg)
    storage_units = n.storage_units.groupby(["nerc_reg", "carrier"]).p_nom.sum().round(0)
    storage_units = storage_units.to_frame(name="Existing")
    existing_energy_capacity = pd.concat([existing_energy_capacity, storage_units])

    # Groupby regions and carriers, then fix indexing
    existing_energy_capacity = existing_energy_capacity.groupby(existing_energy_capacity.index).sum()
    existing_energy_capacity = existing_energy_capacity.reset_index()
    existing_energy_capacity[["Region", "Carrier"]] = pd.DataFrame(
        existing_energy_capacity["index"].tolist(),
        index=existing_energy_capacity.index,
    )
    existing_energy_capacity = existing_energy_capacity.drop(columns="index")
    existing_energy_capacity = existing_energy_capacity.set_index(["Region", "Carrier"])

    # existing_energy_capacity = existing_energy_capacity[existing_energy_capacity.index.get_level_values("Carrier") != "battery"]

    nn_carriers = existing_energy_capacity.index.get_level_values(1).map(n.carriers.nice_name)
    existing_energy_capacity = existing_energy_capacity.droplevel(1)
    existing_energy_capacity = existing_energy_capacity.set_index(nn_carriers, append=True)
    return existing_energy_capacity

def get_statistics(n, column_name):
    """
    Prepare the statistics data for plotting by extracting and grouping by region and carrier.

    Parameters
    ----------
    - n: pypsa.Network
    - column_name: str, the name of the column to extract from statistics (e.g., 'Optimal Capacity', 'Supply')

    Returns
    -------
    - pd.DataFrame: Prepared and grouped data
    """
    groupers = n.statistics.groupers
    df = n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3)
    df = df.loc[["Generator", "StorageUnit", "Link"]]

    # Add nerc_region data
    gens = df.loc["Generator"].index.get_level_values(0)
    gens_reg = gens.map(n.generators.bus.map(n.buses.nerc_reg)).to_series()
    su = df.loc["StorageUnit"].index.get_level_values(0)
    su_reg = su.map(n.storage_units.bus.map(n.buses.nerc_reg)).to_series()
    link = df.loc["Link"].index.get_level_values(0)
    link_reg = link.map(lambda x: n.buses.nerc_reg[n.links.at[x, 'bus1']] 
                     if n.links.at[x, 'carrier'] == "H2 electrolysis" 
                     else n.buses.nerc_reg[n.links.at[x, 'bus0']] 
                     if n.links.at[x, 'carrier'] == "H2 fuel cell" 
                     else n.buses.nerc_reg.get(n.links.at[x, 'bus0'], np.nan)).to_series()
    nerc_reg = pd.concat([gens_reg, su_reg, link_reg])

    df = df.set_index(nerc_reg, append=True)
    df = df.droplevel([0, 1, 2])
    df = df.reset_index()
    df = df.rename(columns={"level_0": "carrier", "level_1": "region"})
    df = df.set_index(["region", "carrier"])

    df_selected = df[column_name]
    df_selected = df_selected.groupby(df_selected.index).sum()
    df_selected = df_selected.reset_index()
    df_selected[["Region", "Carrier"]] = pd.DataFrame(
        df_selected["index"].tolist(),
        index=df_selected.index,
    )
    df_selected = df_selected.drop(columns="index")
    df_selected = df_selected.set_index(["Region", "Carrier"])
    df_selected = df_selected[~df_selected.index.get_level_values("Carrier").isin(["Ac", "Dc"])]

    return df_selected


def get_energy_statistics(n, column_name):
    """
    Prepare the statistics data for plotting by extracting and grouping by region and carrier.

    Parameters
    ----------
    - n: pypsa.Network
    - column_name: str, the name of the column to extract from statistics 
      (e.g., 'Optimal Capacity', 'Supply')

    Returns
    -------
    - pd.DataFrame: Prepared and grouped data
    """

    groupers = n.statistics.groupers
    df = n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3)
    
    # Check if H2 store exists
    has_h2 = not n.stores.empty and "H2" in n.stores.carrier.unique()

    if has_h2:
        df = df.loc[["Store", "StorageUnit"]]
    else:
        # Only StorageUnits if no H2 store exists
        df = df.loc[["StorageUnit"]]

    # --- Map to NERC regions ---
    nerc_reg = []

    # Map stores to regions if present
    if has_h2:
        stores = df.loc["Store"].index.get_level_values(0)
        stores_reg = stores.map(n.stores.bus.map(n.buses.nerc_reg)).to_series()
        nerc_reg.append(stores_reg)

    # Map storage units
    su = df.loc["StorageUnit"].index.get_level_values(0)
    su_reg = su.map(n.storage_units.bus.map(n.buses.nerc_reg)).to_series()
    nerc_reg.append(su_reg)

    # Combine mapping
    nerc_reg = pd.concat(nerc_reg)

    # --- Reformat dataframe ---
    df = df.set_index(nerc_reg, append=True)
    df = df.droplevel([0, 1, 2])
    df = df.reset_index()
    df = df.rename(columns={"level_0": "carrier", "level_1": "region"})
    df = df.set_index(["region", "carrier"])

    # --- Select requested column ---
    df_selected = df[column_name]
    df_selected = df_selected.groupby(df_selected.index).sum()
    df_selected = df_selected.reset_index()

    df_selected[["Region", "Carrier"]] = pd.DataFrame(
        df_selected["index"].tolist(),
        index=df_selected.index,
    )
    df_selected = df_selected.drop(columns="index")
    df_selected = df_selected.set_index(["Region", "Carrier"])

    return df_selected

def plot_bar(data, n, save, title, ylabel, is_capacity=False):
    """
    Plot the data in a bar chart with subplots by region and carrier.

    Parameters
    ----------
    - data: pd.DataFrame, data to plot
    - n: pypsa.Network
    - save: str, file path to save the plot
    - title: str, plot title
    - ylabel: str, y-axis label
    - is_capacity: bool, whether to add extra processing for capacities
    """
    if is_capacity:
        existing_cap = get_currently_installed_capacity(n)
        data.loc[existing_cap.index, "Existing"] = existing_cap
        data = data[["Existing"] + [col for col in data.columns if col != "Existing"]]
        retirements = data.diff(axis=1).clip(upper=0)
        retirements = retirements[(retirements < -0.001).any(axis=1)]
        retirements = retirements.fillna(0)
        data = pd.concat([data, retirements])

    data = data / 1e3  # Convert to GW

    palette = n.carriers.set_index("nice_name").color.to_dict()
    regions = data.index.get_level_values(0).unique()

    # Determine grid layout for subplots
    num_regions = len(regions)
    columns = min(5, num_regions)  # Limit to 5 columns
    rows = math.ceil(num_regions / columns)

    # Set up the figure and axes
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.5, rows * 5),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is a flattened array for consistent indexing
    if num_regions == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

    for i, region in enumerate(regions):
        region_data = data.loc[region]
        region_data.T.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            color=[palette.get(carrier) for carrier in region_data.index.get_level_values(0)],
            legend=False,
        )
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(region)
        axes[i].set_ylabel(ylabel)
        axes[i].set_xlabel("")

    # Remove unused subplots
    if num_regions > 1:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    # Create legend
    handles, labels = [], []
    for carrier in data.reset_index().Carrier.unique():
        handle = plt.Rectangle((0, 0), 1, 1, color=palette[carrier])
        handles.append(handle)
        labels.append(f"{carrier}")
    fig.legend(handles, labels, title="Carrier", loc="lower center", ncol=columns)

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)
    fig.suptitle(title)
    fig.savefig(save)
    plt.close()

def plot_energy_bar(data, n, save, title, ylabel, is_capacity=False):
    """
    Plot the data in a bar chart with subplots by region and carrier.

    Parameters
    ----------
    - data: pd.DataFrame, data to plot
    - n: pypsa.Network
    - save: str, file path to save the plot
    - title: str, plot title
    - ylabel: str, y-axis label
    - is_capacity: bool, whether to add extra processing for capacities
    """
    if is_capacity:
        existing_cap = get_currently_installed_energy_capacity(n)
        data.loc[existing_cap.index, "Existing"] = existing_cap
        data = data[["Existing"] + [col for col in data.columns if col != "Existing"]]
        retirements = data.diff(axis=1).clip(upper=0)
        retirements = retirements[(retirements < -0.001).any(axis=1)]
        retirements = retirements.fillna(0)
        data = pd.concat([data, retirements])

    data = data / 1e3  # Convert to GW

    palette = n.carriers.set_index("nice_name").color.to_dict()
    regions = data.index.get_level_values(0).unique()

    # Determine grid layout for subplots
    num_regions = len(regions)
    columns = min(5, num_regions)  # Limit to 5 columns
    rows = math.ceil(num_regions / columns)

    # Set up the figure and axes
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.5, rows * 5),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is a flattened array for consistent indexing
    if num_regions == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

    for i, region in enumerate(regions):
        region_data = data.loc[region]
        region_data.T.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            color=[palette.get(carrier) for carrier in region_data.index.get_level_values(0)],
            legend=False,
        )
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(region)
        axes[i].set_ylabel(ylabel)
        axes[i].set_xlabel("")

    # Remove unused subplots
    if num_regions > 1:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    # Create legend
    handles, labels = [], []
    for carrier in data.reset_index().Carrier.unique():
        handle = plt.Rectangle((0, 0), 1, 1, color=palette[carrier])
        handles.append(handle)
        labels.append(f"{carrier}")
    fig.legend(handles, labels, title="Carrier", loc="lower center", ncol=columns)

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)
    fig.suptitle(title)
    fig.savefig(save)
    plt.close()

def plot_regional_capacity_additions_bar(n, save):
    """Plot capacity evolution by NERC region in a stacked bar plot."""
    data = get_statistics(n, "Optimal Capacity")
    data2 = get_statistics(n, "Installed Capacity")
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_capacity.csv")
    data2.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_installed_capacity.csv")
    plot_bar(data, n, save, "", "Capacity (GW)", is_capacity=True)

def plot_regional_energy_capacity_additions_bar(n, save):
    """Plot energy capacity evolution by NERC region in a stacked bar plot."""
    data = get_energy_statistics(n, "Optimal Capacity")
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_energy_capacity.csv")
    plot_energy_bar(data, n, save, "", "Energy Capacity (GWh)", is_capacity=True)

def plot_regional_production_bar(n, save):
    """Plot production evolution by NERC region in a stacked bar plot."""
    data = get_statistics(n, "Supply")
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_production.csv")
    plot_bar(data, n, save, "", "Production (GWh)")

def plot_regional_energy_production_bar(n, save):
    """Plot production evolution by NERC region in a stacked bar plot."""
    data = get_energy_statistics(n, "Supply")
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_energy_production.csv")
    plot_energy_bar(data, n, save, "", "Production (GWh)")

def plot_regional_emissions_bar(
    n: pypsa.Network,
    save: str,
) -> None:
    """PLOT OF CO2 EMISSIONS BY NERC REGION AND INVESTMENT PERIOD."""
    regional_emisssions_ts = get_node_emissions_timeseries(n).T.groupby(n.buses.nerc_reg).sum().T / 1e6
    regional_emissions = (
        regional_emisssions_ts.groupby(regional_emisssions_ts.index.get_level_values(0)).sum().round(3).T
    )

    # Determine grid layout for subplots
    regions = regional_emissions.index.get_level_values(0).unique()
    num_regions = len(regions)
    columns = min(5, num_regions)  # Limit to 5 columns
    rows = math.ceil(num_regions / columns)

    # Set up the figure and axes
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.5, rows * 5),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is a flattened array for consistent indexing
    if num_regions == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

    for i, region in enumerate(regions):
        region_data = regional_emissions.loc[region]
        region_data.T.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            legend=False,
        )
        print(region_data) ######################################################
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(region)
        axes[i].set_ylabel("MMtCo2")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)

    plt.xlabel("")
    plt.ylabel("MMtCO2")

    plt.tight_layout()
    plt.savefig(save)
    plt.close()


def plot_emissions_bar(
    n: pypsa.Network,
    save: str,
) -> None:
    """PLOT OF CO2 EMISSIONS BY INVESTMENT PERIOD."""
    emisssions_ts = get_node_emissions_timeseries(n).T.sum().T / 1e6
    emissions = emisssions_ts.groupby(emisssions_ts.index.get_level_values(0)).sum().round(3).T

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(7, 4))
    emissions.T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        legend=False,
    )
    print(emissions) ######################################################
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MMtCo2")
    ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)

    plt.xlabel("CO2 Emissions [MMtCO2]")
    plt.ylabel("MMtCO2")

    plt.tight_layout()
    plt.savefig(save)
    plt.close()


#### Temporal Plots ####


def plot_production_area(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plot timeseries production.

    Will plot an image for the entire time horizon, in addition to
    seperate monthly generation curves
    """

    energy_mix = get_energy_timeseries(n).mul(1e-3)  # MW -> GW
    demand = get_demand_timeseries(n).mul(1e-3)  # MW -> GW

    for carrier in energy_mix.columns:
        if "battery" in carrier or carrier in snakemake.params.electricity["extendable_carriers"]["StorageUnit"]:
            energy_mix[carrier + "_discharger"] = energy_mix[carrier].clip(lower=0.0001)
            energy_mix[carrier + "_charger"] = energy_mix[carrier].clip(upper=-0.0001)
            energy_mix = energy_mix.drop(columns=carrier)
            carriers_2_plot.append(f"{carrier}" + "_charger")
            carriers_2_plot.append(f"{carrier}" + "_discharger")
        if "H2" in carrier or carrier in snakemake.params.electricity["extendable_carriers"]["Store"]:
            energy_mix[carrier + "_discharger"] = energy_mix[carrier].clip(lower=0.0001)
            energy_mix[carrier + "_charger"] = energy_mix[carrier].clip(upper=-0.0001)
            energy_mix = energy_mix.drop(columns=carrier)
            carriers_2_plot.append(f"{carrier}" + "_charger")
            carriers_2_plot.append(f"{carrier}" + "_discharger")
    
    # -------------------------------------------------
    # Remove pure H2 carrier (energy store, not power)
    # -------------------------------------------------

    energy_mix = energy_mix[
        [col for col in energy_mix.columns if col != "H2"]
    ]

    carriers_2_plot = list(set(carriers_2_plot))
    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)

    color_palette = get_color_palette(n)

    months = n.snapshots.get_level_values(1).month.unique()
    num_periods = len(n.investment_periods)
    base_plot_size = 8

    for month in ["all", *months.to_list()]:
        figsize = (14, (base_plot_size * num_periods))
        fig, axs = plt.subplots(figsize=figsize, ncols=1, nrows=num_periods)

        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for i, investment_period in enumerate(n.investment_periods):

            if month == "all":
                sns = n.snapshots[
                    n.snapshots.get_level_values(0) == investment_period
                ]
            else:
                sns = n.snapshots[
                    (n.snapshots.get_level_values(0) == investment_period)
                    & (n.snapshots.get_level_values(1).month == month)
                ]

            # Extract once
            data = energy_mix.loc[sns].droplevel("period").round(2)
            dem = demand.loc[sns].droplevel("period").round(2)

            # Smooth only for full year
            if month == "all":
                data = data.rolling(56, center=True, min_periods=1).mean()
                dem = dem.rolling(56, center=True, min_periods=1).mean()
            
            # -------------------------------------------------
            # Apply same stack order as weekly energy plot
            # -------------------------------------------------

            positive_order = [
                "Nuclear",
                "Coal",
                "Oil",
                "Combined-Cycle Gas",
                "Open-Cycle Gas",
                "Ccgt-95Ccs", 
                "Reservoir & Dam",
                "Biomass",
                "Waste",
                "Geothermal",
                "battery_discharger",
                "4hr_battery_storage_discharger",
                "8hr_battery_storage_discharger",
                "H2 electrolysis_discharger",
                "100hr_battery_storage_discharger",
                "Solar",
                "Floating Offshore Wind",
                "Onshore Wind",
            ]

            negative_order = [
                "battery_charger",
                "4hr_battery_storage_charger",
                "8hr_battery_storage_charger",
                "H2 fuel cell_charger",
                "100hr_battery_storage_charger",
            ]

            # Keep only carriers that actually exist in the dataframe
            neg_existing = [c for c in negative_order if c in data.columns]
            pos_existing = [c for c in positive_order if c in data.columns]

            # Optional: catch anything you forgot
            remaining = [c for c in data.columns if c not in neg_existing + pos_existing]

            # Final stacking order
            data = data[neg_existing + pos_existing + remaining]

            # Plot once
            data.plot.area(
                ax=axs[i],
                alpha=0.85,
                color=color_palette,
                linewidth=0
            )

            dem.plot(
                ax=axs[i],
                ls="--",
                color="black",
                linewidth=3,
            )

            suffix = "-" + datetime.strptime(str(month), "%m").strftime("%b") if month != "all" else ""

            axs[i].legend(bbox_to_anchor=(1, 1), loc="upper left")
            axs[i].set_ylabel("Power [GW]")
            axs[i].set_xlabel("")
            axs[i].set_xlim(pd.Timestamp(f"{investment_period}-01-01"), pd.Timestamp(f"{investment_period+1}-01-01"))
            axs[i].margins(x=0)

            # Monthly ticks
            axs[i].xaxis.set_major_locator(mdates.MonthLocator())

            # Format as abbreviated month name
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(create_title("Production [GW]", **wildcards))

        save = Path(save)
        fig.savefig(save.parent / (save.stem + suffix + save.suffix))
        plt.close()


def plot_hourly_emissions(n: pypsa.Network, save: str, **wildcards) -> None:
    """Plots snapshot emissions by technology."""
    # get data
    emissions = get_tech_emissions_timeseries(n).mul(1e-6)  # T -> MT
    zeros = emissions.columns[(np.abs(emissions) < 1e-7).all()]
    emissions = emissions.drop(columns=zeros)

    # plot
    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    if not emissions.empty:
        emissions.plot.area(
            ax=ax,
            alpha=0.7,
            legend="reverse",
            color=color_palette,
        )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Technology Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)
    plt.close()


def plot_accumulated_emissions_tech(n: pypsa.Network, save: str, **wildcards) -> None:
    """Creates area plot of accumulated emissions by technology."""
    # get data

    emissions = get_tech_emissions_timeseries(n).cumsum().mul(1e-6)  # T -> MT
    zeros = emissions.columns[(np.abs(emissions) < 1e-7).all()]
    emissions = emissions.drop(columns=zeros)

    # plot

    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    if not emissions.empty:
        emissions.plot.area(
            ax=ax,
            alpha=0.7,
            legend="reverse",
            color=color_palette,
        )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Technology Accumulated Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)
    plt.close()


def plot_accumulated_emissions(n: pypsa.Network, save: str, **wildcards) -> None:
    """Plots accumulated emissions."""
    # get data

    emissions = get_tech_emissions_timeseries(n).mul(1e-6).sum(axis=1)  # T -> MT
    emissions = emissions.cumsum().to_frame("co2")

    # plot

    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))

    emissions.plot.area(
        ax=ax,
        alpha=0.7,
        legend="reverse",
        color=color_palette,
    )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Accumulated Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def plot_capacity_factor_heatmap(n: pypsa.Network, save: str, **wildcards) -> None:
    """HEATMAP OF RENEWABLE CAPACITY FACTORS BY CARRIER."""
    df_long = n.generators_t.p.loc[n.investment_periods[0]].melt(
        var_name="bus",
        value_name="p",
        ignore_index=False,
    )
    df_long["region"] = df_long["bus"].map(n.generators.bus.map(n.buses.country))
    df_long["carrier"] = df_long["bus"].map(n.generators.carrier)
    df_long["hour"] = df_long.index.hour
    df_long["month"] = df_long.index.month
    df_long = df_long.drop(columns="bus")
    df_long = df_long.drop(columns="region").groupby(["carrier", "month", "hour"]).mean().reset_index()

    # Get unique months for separate panels
    unique_months = df_long["month"].unique()

    # Prepare figure and axes
    fig, axs = plt.subplots(
        len(unique_months),
        1,
        figsize=(12, len(unique_months) * 4),
        sharex=True,
    )

    # Iterate over each month to create a panel
    for idx, month in enumerate(sorted(unique_months)):
        month_data = df_long[df_long["month"] == month]
        pivot_data = month_data.pivot(index="hour", columns="carrier", values="p")

        ax = axs[idx] if len(unique_months) > 1 else axs
        pivot_data.plot.area(ax=ax, title=f"Month: {month}", alpha=0.7)
        ax.set_ylabel("Mean Power (p)")
        ax.set_xlabel("Hour of the Day")
        ax.legend(title="Carrier", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle("Heatmap of Renewable Capacity Factors by by Carrier")
    plt.tight_layout()

    plt.savefig(save)
    plt.close()


#### Panel / Mixed Plots ####


def plot_generator_data_panel(
    n: pypsa.Network,
    save: str,
    **wildcards,
):
    df_capex_expand = n.generators.loc[
        n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing"),
        :,
    ]

    df_storage_units = n.storage_units.loc[n.storage_units.p_nom_extendable, :].copy()
    df_storage_units.loc[:, "efficiency"] = df_storage_units.efficiency_dispatch
    df_capex_expand = pd.concat([df_capex_expand, df_storage_units])

    df_efficiency = n.generators.loc[
        ~n.generators.carrier.isin(
            ["solar", "onwind", "offwind", "offwind_floating", "hydro", "load"],
        ),
        :,
    ]
    # Create a figure and subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Plot on each subplot
    sns.lineplot(
        data=get_generator_marginal_costs(n),
        x="timestep",
        y="Value",
        hue="Carrier",
        ax=axes[0, 0],
    )
    sns.barplot(data=df_capex_expand, x="carrier", y="capital_cost", ax=axes[0, 1])
    sns.boxplot(data=df_efficiency, x="carrier", y="efficiency", ax=axes[1, 0])

    # Create line plot of declining capital costs
    sns.lineplot(
        data=df_capex_expand[df_capex_expand.build_year > 0],
        x="build_year",
        y="capital_cost",
        hue="carrier",
        ax=axes[2, 0],
    )

    cf_profiles = n.get_switchable_as_dense("Generator", "p_max_pu")
    fuel_costs = n.generators.marginal_cost * cf_profiles.sum()
    n.generators["lcoe"] = (n.generators.capital_cost + fuel_costs) / cf_profiles.sum()
    n.generators["cf"] = cf_profiles.mean()
    lcoe_plot_df = n.generators.loc[
        n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing"),
        :,
    ]

    sns.boxplot(
        data=n.generators,
        x="cf",
        y="carrier",
        ax=axes[1, 1],
    )

    sns.boxplot(
        data=lcoe_plot_df,
        x="lcoe",
        y="carrier",
        ax=axes[2, 1],
    )

    # Set titles for each subplot
    axes[0, 0].set_title("Generator Marginal Costs")
    axes[0, 1].set_title("Extendable Capital Costs")
    axes[1, 0].set_title("Plant Efficiency")
    axes[1, 1].set_title("Capacity Factors by Carrier")
    axes[2, 0].set_title("Expansion Capital Costs by Carrier")
    axes[2, 1].set_title("LCOE by Carrier")

    # Set labels for each subplot
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("$ / MWh")
    # axes[0, 0].set_ylim(0, 200)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("$ / MW-yr")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("MWh_primary / MWh_elec")
    axes[1, 1].set_xlabel("p.u.")
    axes[1, 1].set_ylabel("")
    axes[2, 0].set_xlabel("Year")
    axes[2, 0].set_ylabel("$ / MW-yr")
    axes[2, 1].set_xlabel("$ / MWh")
    axes[2, 1].set_ylabel("")

    # Rotate x-axis labels for each subplot
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)

    # Lay legend out horizontally
    axes[0, 0].legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        fontsize="xx-small",
    )
    axes[2, 0].legend(fontsize="xx-small")

    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def plot_region_lmps(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """Plots a box plot of LMPs for each region."""
    df_lmp = n.buses_t.marginal_price
    df_long = pd.melt(
        df_lmp.reset_index(),
        id_vars=["timestep"],
        var_name="bus",
        value_name="lmp",
    )
    df_long["season"] = df_long["timestep"].dt.quarter
    df_long["hour"] = df_long["timestep"].dt.hour
    df_long = df_long.drop(columns="timestep")
    df_long["region"] = df_long.bus.map(n.buses.country)

    plt.figure(figsize=(10, 10))

    sns.boxplot(
        df_long,
        x="lmp",
        y="region",
        width=0.5,
        fliersize=0.5,
        linewidth=1,
    )

    plt.title(create_title("LMPs by Region", **wildcards))
    plt.xlabel("LMP [$/MWh]")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(save)
    plt.close()


# Fuel costs


def plot_fuel_costs(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    fuel_costs = get_fuel_costs(n)

    fuels = set(fuel_costs.index.get_level_values("carrier"))

    fig, axs = plt.subplots(len(fuels) + 1, 1, figsize=(20, 40))

    color_palette = n.carriers.color.to_dict()

    # plot error plot of all fuels
    df = fuel_costs.droplevel(["bus", "Generator"]).T.resample("d").mean().reset_index().melt(id_vars="timestep")
    sns.lineplot(
        data=df,
        x="timestep",
        y="value",
        hue="carrier",
        ax=axs[0],
        legend=True,
        palette=color_palette,
    )
    (axs[0].set_title("Daily Average Fuel Costs [$/MWh]"),)
    (axs[0].set_xlabel(""),)
    (axs[0].set_ylabel("$/MWh"),)

    # plot bus fuel prices for each fuel
    for i, fuel in enumerate(fuels):
        nice_name = n.carriers.at[fuel, "nice_name"]
        df = fuel_costs.loc[fuel, :, :].droplevel("Generator").T.resample("d").mean().T.groupby(level=0).mean().T
        sns.lineplot(
            data=df,
            legend=False,
            palette="muted",
            dashes=False,
            ax=axs[i + 1],
        )
        (axs[i + 1].set_title(f"Daily Average {nice_name} Fuel Costs per Bus [$/MWh]"),)
        (axs[i + 1].set_xlabel(""),)
        (axs[i + 1].set_ylabel("$/MWh"),)

    fig.savefig(save)
    plt.close()


# def plot_storage_soc(n: pypsa.Network, output_path: str, **wildcards) -> None:
#     """
#     Plot hourly state of charge (SOC) of extendable
#     100hr battery storage units, aggregated by NERC region using
#     capacity-weighted averaging of normalized SOC (0–1).
#     """

#     # -------------------------------------------------
#     # Select extendable 100hr batteries
#     # -------------------------------------------------

#     su = n.storage_units[
#         (n.storage_units.carrier == "100hr_battery_storage")
#         & (n.storage_units.p_nom_extendable)
#     ]

#     if su.empty:
#         logger.warning("No extendable 100hr_battery_storage units found.")
#         return

#     su = su.copy()

#     # -------------------------------------------------
#     # Map storage units to NERC region
#     # -------------------------------------------------

#     if "nerc_reg" not in n.buses.columns:
#         raise ValueError("No 'nerc_reg' column found in buses.")
#     # #here
#     # # -------------------------------------------------
#     # # Inspect available spatial columns in buses
#     # # -------------------------------------------------

#     # print("\nAvailable columns in n.buses:")
#     # print(n.buses.columns.tolist())

#     # print("\nColumns with few unique values (likely spatial groupings):")
#     # for col in n.buses.columns:
#     #     nunique = n.buses[col].nunique(dropna=True)
#     #     if 1 < nunique < 100:  # avoid continuous variables like x/y
#     #         print(f"{col}: {nunique} unique values")
    
#     # print("\n================ n.buses['nerc_reg'] ==================")

#     # pd.set_option("display.max_rows", None)
#     # pd.set_option("display.max_columns", None)
#     # pd.set_option("display.width", None)

#     # print(n.buses["nerc_reg"])

#     # print("\n================ n.buses['country'] ==================")

#     # pd.set_option("display.max_rows", None)
#     # pd.set_option("display.max_columns", None)
#     # pd.set_option("display.width", None)

#     # print(n.buses["country"])
#     # #to here
#     su["nerc_reg"] = su.bus.map(n.buses.nerc_reg)
#     su = su.dropna(subset=["nerc_reg"])

#     if su.empty:
#         logger.warning("No storage units with valid NERC region mapping.")
#         return

#     # -------------------------------------------------
#     # Capacity-weighted SOC aggregation (normalized 0–1)
#     # -------------------------------------------------

#     energy_capacity = su.p_nom_opt * su.max_hours  # MWh per unit
#     soc_mwh = n.storage_units_t.state_of_charge[su.index]  # MWh time series

#     region_soc = {}
#     region_energy_capacity = {}
#     region_p_nom = {}

#     for region, group in su.groupby("nerc_reg"):

#         units = group.index

#         weights = energy_capacity.loc[units]  # MWh
#         total_energy_cap = weights.sum()

#         total_p_nom = group.p_nom_opt.sum()

#         region_energy_capacity[region] = total_energy_cap
#         region_p_nom[region] = total_p_nom

#         # SAFETY: if installed power < 100 MW → treat as zero
#         if total_p_nom < 100:
#             region_soc[region] = pd.Series(
#                 0.0, index=soc_mwh.index
#             )
#             continue

#         weighted_energy = soc_mwh[units].multiply(weights, axis=1).sum(axis=1)

#         soc_fraction = weighted_energy / total_energy_cap

#         region_soc[region] = soc_fraction

#     region_soc = pd.DataFrame(region_soc)
#     # #here
#     # # -------------------------------------------------
#     # # FULL DEBUG PRINTS
#     # # -------------------------------------------------

#     # import pandas as pd

#     # pd.set_option("display.max_rows", None)
#     # pd.set_option("display.max_columns", None)
#     # pd.set_option("display.width", None)

#     # print("\n================ ENERGY CAPACITY (MWh) ==================")
#     # print(pd.Series(region_energy_capacity))

#     # print("\n================ INSTALLED POWER (MW) ==================")
#     # print(pd.Series(region_p_nom))

#     # print("\n================ SOC_MWH (raw state_of_charge) ==================")
#     # print(soc_mwh)

#     # print("\n================ REGION SOC FRACTION (0–1) ==================")
#     # print(region_soc)
#     # #to here
#     # -------------------------------------------------
#     # Handle PyPSA MultiIndex snapshots safely
#     # -------------------------------------------------

#     if isinstance(region_soc.index, pd.MultiIndex):
#         time_index = region_soc.index.get_level_values(-1)
#     else:
#         time_index = region_soc.index

#     region_soc.index = pd.to_datetime(time_index)

#     # -------------------------------------------------
#     # Rename regions for cleaner legend
#     # -------------------------------------------------

#     region_soc = region_soc.rename(columns={
#         "WECC_NWPP": "NWPP",
#         "WECC_CA": "CA",
#         "WECC_SRSG": "SRSG"
#     })

#     # -------------------------------------------------
#     # Plot with custom colors
#     # -------------------------------------------------

#     colors = {
#         "NWPP": "darkgreen",
#         "CA": "blue",
#         "SRSG": "orange"
#     }

#     fig, ax = plt.subplots(figsize=(12, 6))

#     for region in region_soc.columns:
#         ax.plot(
#             region_soc.index,
#             region_soc[region],
#             label=region,
#             color=colors.get(region, None),
#         )

#     ax.set_ylabel("State of Charge (fraction of capacity)")
#     # ax.set_ylim(0, 1)
#     ax.set_title("Hourly SOC of Metal-Air Batteries by NERC Region")

#     ax.legend(title=None, bbox_to_anchor=(1.05, 1), loc="upper left")

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=300)
#     plt.close(fig)

#     logger.info(f"Saved SOC plot to {output_path}")

def plot_storage_soc(n: pypsa.Network, output_path: str, **wildcards) -> None:
    """
    Plot hourly normalized SOC (0–1) of extendable
    100hr battery storage units per country.

    One page per country in a multi-page PDF.
    Only countries with >= 100 MW installed metal-air capacity are plotted.
    """

    if not (n.storage_units.carrier == "100hr_battery_storage").any():
        print("No LDES")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_axis_off()
        ax.set_title("No 100hr battery storage in this scenario")

        fig.savefig(output_path)
        plt.close(fig)

        return

    # -------------------------------------------------
    # Select extendable 100hr batteries
    # -------------------------------------------------

    su = n.storage_units[
        (n.storage_units.carrier == "100hr_battery_storage")
        & (n.storage_units.p_nom_extendable)
    ]

    if su.empty:
        logger.warning("No extendable 100hr_battery_storage units found.")
        return

    su = su.copy()

    # -------------------------------------------------
    # Map storage units to country and NERC region
    # -------------------------------------------------

    if "country" not in n.buses.columns:
        raise ValueError("No 'country' column found in buses.")

    su["country"] = su.bus.map(n.buses.country)

    if "nerc_reg" in n.buses.columns:
        su["nerc_reg"] = su.bus.map(n.buses.nerc_reg)
    else:
        su["nerc_reg"] = "Unknown"

    su = su.dropna(subset=["country"])

    if su.empty:
        logger.warning("No storage units with valid country mapping.")
        return

    # -------------------------------------------------
    # Get SOC time series
    # -------------------------------------------------

    soc_mwh = n.storage_units_t.state_of_charge[su.index]

    if isinstance(soc_mwh.index, pd.MultiIndex):
        time_index = soc_mwh.index.get_level_values(-1)
    else:
        time_index = soc_mwh.index

    soc_mwh.index = pd.to_datetime(time_index)

    # -------------------------------------------------
    # Plot per country (multi-page PDF)
    # -------------------------------------------------

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:

        for country, group in su.groupby("country"):

            total_p_nom = group.p_nom_opt.sum()

            if total_p_nom < 100:
                continue

            units = group.index

            # Total installed energy capacity (MWh)
            total_energy_cap = (group.p_nom_opt * group.max_hours).sum()

            # Aggregate SOC in MWh (no weighting needed now)
            soc_country_mwh = soc_mwh[units].sum(axis=1)

            # Normalize to 0–1
            soc_fraction = soc_country_mwh / total_energy_cap

            # -------------------------------------------------
            # Get unique NERC region(s)
            # -------------------------------------------------

            nerc_regions = group["nerc_reg"].dropna().unique()
            nerc_str = ", ".join(nerc_regions)

            # -------------------------------------------------
            # Plot
            # -------------------------------------------------

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.fill_between(
                soc_fraction.index,
                0,
                soc_fraction,
                color="#b800b8",
                alpha=0.5
            )

            ax.plot(
                soc_fraction.index,
                soc_fraction,
                color="#b800b8",
                linewidth=1.5
            )

            ax.set_ylabel("State of Charge (fraction of capacity)")
            # No ylim — you want to inspect bounds

            ax.set_title(
                f"Hourly SOC of Metal-Air Batteries — {country}\n"
                f"NERC: {nerc_str} | Installed capacity: {total_p_nom:.0f} MW"
            )

            ax.set_xlim(pd.Timestamp("2040-01-01"), pd.Timestamp("2041-01-01"))
            ax.set_ylim(0, 1)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logger.info(f"Saved SOC plots to {output_path}")


def _align_zero_axes(ax1: plt.Axes, ax2: plt.Axes) -> None:
    """
    Align the zero lines of two y-axes sharing the same x-axis.
    Expands limits symmetrically so zero sits at the same vertical
    position on both axes without clipping any data.
    """
    def get_zero_ratio(ax):
        lo, hi = ax.get_ylim()
        total = hi - lo
        return (-lo / total) if total != 0 else 0.5

    # Get data ranges
    lo1, hi1 = ax1.get_ylim()
    lo2, hi2 = ax2.get_ylim()

    # Fraction of axis height that sits below zero on ax1
    ratio = -lo1 / (hi1 - lo1) if (hi1 - lo1) != 0 else 0.5

    # Rescale ax2 so its zero sits at the same fraction
    # ax2 only has non-negative data (wind CF), so lo2 = 0
    # We need: 0 - new_lo2 / (new_hi2 - new_lo2) = ratio
    # => new_lo2 = -ratio * new_hi2 / (1 - ratio)
    new_lo2 = -ratio * hi2 / (1 - ratio) if ratio < 1 else -hi2
    ax2.set_ylim(new_lo2, hi2)


# def plot_storage_soc_and_wind_cf(n: pypsa.Network, output_path: str, **wildcards) -> None:
#     """
#     Plot hourly normalized SOC (0–1) of extendable 100hr battery storage units
#     AND the onshore wind capacity factor during the same period, per country.

#     One page per country in a multi-page PDF.
#     Only countries with >= 100 MW installed metal-air capacity are plotted.
#     Two vertically stacked subplots per page:
#       - Top:    Metal-air battery SOC (fraction of capacity)
#       - Bottom: Onshore wind capacity factor (P_actual / P_nom_opt)
#     """

#     if not (n.storage_units.carrier == "100hr_battery_storage").any():
#         print("No LDES")

#         fig, ax = plt.subplots(figsize=(12, 6))
#         ax.set_axis_off()
#         ax.set_title("No 100hr battery storage in this scenario")
#         fig.savefig(output_path)
#         plt.close(fig)
#         return

#     # -------------------------------------------------
#     # Select extendable 100hr batteries
#     # -------------------------------------------------

#     su = n.storage_units[
#         (n.storage_units.carrier == "100hr_battery_storage")
#         & (n.storage_units.p_nom_extendable)
#     ].copy()

#     if su.empty:
#         logger.warning("No extendable 100hr_battery_storage units found.")
#         return

#     # -------------------------------------------------
#     # Map storage units to country and NERC region
#     # -------------------------------------------------

#     if "country" not in n.buses.columns:
#         raise ValueError("No 'country' column found in buses.")

#     su["country"] = su.bus.map(n.buses.country)

#     if "nerc_reg" in n.buses.columns:
#         su["nerc_reg"] = su.bus.map(n.buses.nerc_reg)
#     else:
#         su["nerc_reg"] = "Unknown"

#     su = su.dropna(subset=["country"])

#     if su.empty:
#         logger.warning("No storage units with valid country mapping.")
#         return

#     # -------------------------------------------------
#     # Get SOC time series
#     # -------------------------------------------------

#     soc_mwh = n.storage_units_t.state_of_charge[su.index].copy()

#     if isinstance(soc_mwh.index, pd.MultiIndex):
#         time_index = soc_mwh.index.get_level_values(-1)
#     else:
#         time_index = soc_mwh.index

#     soc_mwh.index = pd.to_datetime(time_index)

#     # -------------------------------------------------
#     # Build onshore wind capacity factor per country
#     # -------------------------------------------------

#     gens = n.generators[n.generators.carrier == "onwind"].copy()

#     if gens.empty:
#         logger.warning("No onshore wind generators found — wind CF subplot will be blank.")
#         wind_cf_by_country = {}
#     else:
#         gens["country"] = gens.bus.map(n.buses.country)
#         gens = gens.dropna(subset=["country"])

#         target_period = n.investment_periods[-1]  # e.g. 2040

#         # Use the same snapshot selection pattern as plot_production_area
#         sns = n.snapshots[n.snapshots.get_level_values(0) == target_period]

#         # Per-generator dispatch for the target period, drop period level
#         wind_dispatch = (
#             n.generators_t.p[gens.index]
#             .loc[sns]
#             .droplevel("period")
#         )
#         wind_dispatch.index = pd.to_datetime(wind_dispatch.index)

#         wind_cf_by_country = {}
#         for country, grp in gens.groupby("country"):
#             total_nom = grp.p_nom_opt.sum()
#             if total_nom <= 0:
#                 continue
#             # Sum per-generator dispatch for this country, then normalise
#             wind_cf_by_country[country] = wind_dispatch[grp.index].sum(axis=1) / total_nom

#         print("=== GENERATORS_T.P DEBUG ===")
#         target_period = n.investment_periods[-1]
#         sns = n.snapshots[n.snapshots.get_level_values(0) == target_period]

#         raw = n.generators_t.p[gens.index]
#         print("raw index type:", type(raw.index))
#         print("raw index sample:", raw.index[:3].tolist())
#         print("sns sample:", sns[:3].tolist())

#         sliced = raw.loc[sns]
#         print("sliced shape:", sliced.shape)
#         print("sliced sum per generator:\n", sliced.sum())

#         dropped = sliced.droplevel("period")
#         print("dropped index sample:", dropped.index[:3].tolist())
#         print("dropped value range:", dropped.min().min(), "to", dropped.max().max())

#         # Per country sum
#         for country, grp in gens.groupby("country"):
#             total_nom = grp.p_nom_opt.sum()
#             dispatch = dropped[grp.index].sum(axis=1)
#             print(f"{country}: p_nom_opt={total_nom:.1f}, dispatch max={dispatch.max():.2f}, CF max={dispatch.max()/total_nom:.4f}")

#     # -------------------------------------------------
#     # Plot per country (multi-page PDF)
#     # -------------------------------------------------

#     from matplotlib.backends.backend_pdf import PdfPages

#     with PdfPages(output_path) as pdf:

#         for country, group in su.groupby("country"):

#             total_p_nom = group.p_nom_opt.sum()
#             if total_p_nom < 100:
#                 continue

#             # ---- SOC ----
#             total_energy_cap = (group.p_nom_opt * group.max_hours).sum()
#             soc_country_mwh = soc_mwh[group.index].sum(axis=1)
#             soc_fraction = soc_country_mwh / total_energy_cap

#             print("\n=== SOC FRACTION DEBUG ===")
#             print("soc_fraction index sample:", soc_fraction.index[:5])

#             nerc_regions = group["nerc_reg"].dropna().unique()
#             nerc_str = ", ".join(nerc_regions)

#             # ---- Wind CF ----
#             wind_cf = wind_cf_by_country.get(country)  # None if no wind in country

#             # ---- Figure: two stacked subplots ----
#             fig, (ax_soc, ax_cf) = plt.subplots(
#                 nrows=2, ncols=1,
#                 figsize=(12, 10),
#                 sharex=True,
#             )

#             # -- Top: SOC --
#             ax_soc.fill_between(
#                 soc_fraction.index, 0, soc_fraction,
#                 color="#b800b8", alpha=0.5,
#             )
#             ax_soc.plot(
#                 soc_fraction.index, soc_fraction,
#                 color="#b800b8", linewidth=1.5,
#             )
#             ax_soc.set_ylabel("State of Charge\n(fraction of capacity)")
#             ax_soc.set_ylim(0, 1)
#             ax_soc.set_title(
#                 f"Metal-Air Battery SOC & Onshore Wind CF — {country}\n"
#                 f"NERC: {nerc_str} | Battery installed: {total_p_nom:.0f} MW"
#             )

#             # -- Bottom: Wind CF --
#             if wind_cf is not None:
#                 # Align to the same time window as SOC
#                 wind_cf_plot = wind_cf.reindex(soc_fraction.index)

#                 ax_cf.fill_between(
#                     wind_cf_plot.index, 0, wind_cf_plot,
#                     color="#f5a623", alpha=0.5,
#                 )
#                 ax_cf.plot(
#                     wind_cf_plot.index, wind_cf_plot,
#                     color="#f5a623", linewidth=1.0,
#                 )
#                 ax_cf.set_ylabel("Onshore Wind\nCapacity Factor")
#                 ax_cf.set_ylim(0, 1)
#             else:
#                 ax_cf.set_axis_off()
#                 ax_cf.text(
#                     0.5, 0.5, "No onshore wind data for this country",
#                     ha="center", va="center", transform=ax_cf.transAxes,
#                 )
#                 ax_cf.set_ylabel("Onshore Wind\nCapacity Factor")

#             ax_cf.set_xlim(pd.Timestamp("2040-01-01"), pd.Timestamp("2041-01-01"))
#             ax_cf.set_xlabel("Date")

#             fig.tight_layout()
#             pdf.savefig(fig)
#             plt.close(fig)

#     logger.info(f"Saved SOC + wind CF plots to {output_path}")

def plot_storage_soc_and_wind_cf(n: pypsa.Network, output_path: str, **wildcards) -> None:
    """
    Plot hourly normalized SOC (0–1) of extendable 100hr battery storage units
    AND the onshore wind capacity factor during the same period, per country.

    One page per country in a multi-page PDF.
    Only countries with >= 100 MW installed metal-air capacity are plotted.
    Two vertically stacked subplots per page:
      - Top:    Metal-air battery SOC (fraction of capacity) — raw fill + line
      - Bottom: Onshore wind capacity factor — raw fill + 14-day rolling mean overlay
    """

    if not (n.storage_units.carrier == "100hr_battery_storage").any():
        print("No LDES")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_axis_off()
        ax.set_title("No 100hr battery storage in this scenario")
        fig.savefig(output_path)
        plt.close(fig)
        return

    # -------------------------------------------------
    # Select extendable 100hr batteries
    # -------------------------------------------------

    su = n.storage_units[
        (n.storage_units.carrier == "100hr_battery_storage")
        & (n.storage_units.p_nom_extendable)
    ].copy()

    if su.empty:
        logger.warning("No extendable 100hr_battery_storage units found.")
        return

    # -------------------------------------------------
    # Map storage units to country and NERC region
    # -------------------------------------------------

    if "country" not in n.buses.columns:
        raise ValueError("No 'country' column found in buses.")

    su["country"] = su.bus.map(n.buses.country)

    if "nerc_reg" in n.buses.columns:
        su["nerc_reg"] = su.bus.map(n.buses.nerc_reg)
    else:
        su["nerc_reg"] = "Unknown"

    su = su.dropna(subset=["country"])

    if su.empty:
        logger.warning("No storage units with valid country mapping.")
        return

    # -------------------------------------------------
    # Get SOC time series
    # -------------------------------------------------

    soc_mwh = n.storage_units_t.state_of_charge[su.index].copy()

    if isinstance(soc_mwh.index, pd.MultiIndex):
        time_index = soc_mwh.index.get_level_values(-1)
    else:
        time_index = soc_mwh.index

    soc_mwh.index = pd.to_datetime(time_index)

    # -------------------------------------------------
    # Build onshore wind capacity factor per country
    # -------------------------------------------------

    gens = n.generators[n.generators.carrier == "onwind"].copy()

    if gens.empty:
        logger.warning("No onshore wind generators found — wind CF subplot will be blank.")
        wind_cf_by_country = {}
    else:
        gens["country"] = gens.bus.map(n.buses.country)
        gens = gens.dropna(subset=["country"])

        target_period = n.investment_periods[-1]  # e.g. 2040
        sns = n.snapshots[n.snapshots.get_level_values(0) == target_period]

        wind_dispatch = (
            n.generators_t.p[gens.index]
            .loc[sns]
            .droplevel("period")
        )
        wind_dispatch.index = pd.to_datetime(wind_dispatch.index)

        wind_cf_by_country = {}
        for country, grp in gens.groupby("country"):
            total_nom = grp.p_nom_opt.sum()
            if total_nom <= 0:
                continue
            wind_cf_by_country[country] = wind_dispatch[grp.index].sum(axis=1) / total_nom

    # -------------------------------------------------
    # Rolling window: 14 days at 3h resolution = 14 × 8 = 112 timesteps
    # -------------------------------------------------

    ROLLING_WIN = 14 * 8

    # -------------------------------------------------
    # Plot per country (multi-page PDF)
    # -------------------------------------------------

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:

        for country, group in su.groupby("country"):

            total_p_nom = group.p_nom_opt.sum()
            if total_p_nom < 100:
                if country not in wind_cf_by_country:
                    continue

            # ---- SOC ----
            total_energy_cap = (group.p_nom_opt * group.max_hours).sum()
            soc_country_mwh = soc_mwh[group.index].sum(axis=1)
            soc_fraction = soc_country_mwh / total_energy_cap

            nerc_regions = group["nerc_reg"].dropna().unique()
            nerc_str = ", ".join(nerc_regions)

            wind_cf = wind_cf_by_country.get(country)

            # ---- Figure: two stacked subplots ----
            fig, (ax_soc, ax_cf) = plt.subplots(
                nrows=2, ncols=1,
                figsize=(12, 10),
                sharex=True,
            )

            # -- Top: SOC — original raw fill + line, no rolling mean --
            if total_p_nom >= 100:
                ax_soc.fill_between(
                    soc_fraction.index, 0, soc_fraction,
                    color="#b800b8", alpha=0.5,
                )
                ax_soc.plot(
                    soc_fraction.index, soc_fraction,
                    color="#b800b8", linewidth=1.5,
                )
                ax_soc.set_ylim(0, 1)
            else:
                ax_soc.set_axis_off()
                ax_soc.text(
                    0.5, 0.5, "No significant battery storage in this country",
                    ha="center", va="center", transform=ax_soc.transAxes,
                )

            ax_soc.set_ylabel("State of Charge\n(fraction of capacity)")
            ax_soc.set_title(
                f"Metal-Air Battery SOC & Onshore Wind CF — {country}\n"
                f"NERC: {nerc_str} | Battery installed: {total_p_nom:.0f} MW"
            )

            # -- Bottom: Wind CF — raw fill + 14-day rolling mean overlay --
            if wind_cf is not None:
                wind_cf_plot = wind_cf.reindex(soc_fraction.index)
                wind_cf_smooth = wind_cf_plot.rolling(ROLLING_WIN, center=True, min_periods=1).mean()

                ax_cf.fill_between(
                    wind_cf_plot.index, 0, wind_cf_plot,
                    color="#235ebc", alpha=0.2, linewidth=0,
                )
                ax_cf.plot(
                    wind_cf_plot.index, wind_cf_plot,
                    color="#235ebc", linewidth=0.5, alpha=0.4, label="Raw CF",
                )
                ax_cf.plot(
                    wind_cf_smooth.index, wind_cf_smooth,
                    color="#235ebc", linewidth=2.5, alpha=1.0, label="14-day mean",
                )
                ax_cf.set_ylim(0, 1)
                ax_cf.legend(loc="upper right", fontsize=9)
            else:
                ax_cf.set_axis_off()
                ax_cf.text(
                    0.5, 0.5, "No onshore wind data for this country",
                    ha="center", va="center", transform=ax_cf.transAxes,
                )

            ax_cf.set_ylabel("Onshore Wind\nCapacity Factor")
            ax_cf.set_xlim(pd.Timestamp("2040-01-01"), pd.Timestamp("2041-01-01"))
            ax_cf.set_xlabel("Date")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logger.info(f"Saved SOC + wind CF plots to {output_path}")



def plot_market_price(n: pypsa.Network, output_path: str, **wildcards) -> None:
    """
    Plot load-weighted average market price ($/MWh) over time.

    One subplot per investment period. Automatically smooths
    hourly data with a 24-hour rolling mean for clarity.
    """

    base_plot_size = 6
    num_periods = len(n.investment_periods)

    # --- Load-weighted average price ---
    prices = n.buses_t.marginal_price  # $/MWh
    loads = n.loads_t.p_set

    # Aggregate loads to buses
    load_bus = n.loads.bus
    load_by_bus = loads.groupby(load_bus, axis=1).sum()

    # Weighted average price
    weighted_price = (prices * load_by_bus).sum(axis=1) / load_by_bus.sum(axis=1)

    # # --- Diagnostic: inspect snapshot resolution ---
    # snapshot_index = weighted_price.index
    # if hasattr(snapshot_index, 'get_level_values'):
    #     # Multi-index (investment periods present)
    #     raw_timestamps = snapshot_index.get_level_values(-1)
    # else:
    #     raw_timestamps = snapshot_index

    # if len(raw_timestamps) > 1:
    #     inferred_freq = pd.infer_freq(raw_timestamps[:50])  # sample first 50 to be safe
    #     typical_delta = raw_timestamps[1] - raw_timestamps[0]
    #     logger.info(
    #         f"Snapshot resolution diagnostic — "
    #         f"first timestamp: {raw_timestamps[0]}, "
    #         f"typical delta: {typical_delta}, "
    #         f"inferred freq: {inferred_freq}"
    #     )

    # --- Figure setup ---
    figsize = (14, base_plot_size * num_periods)
    fig, axs = plt.subplots(nrows=num_periods, ncols=1, figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, period in enumerate(n.investment_periods):
        # select snapshots for this investment period
        sns = weighted_price.index[weighted_price.index.get_level_values(0) == period]
        price_series = weighted_price.loc[sns].droplevel("period")

        # reconstruct what a full continuous index should look like
        full_index = pd.date_range(
            start=n.snapshots.get_level_values(1)[0],
            end=n.snapshots.get_level_values(1)[-1],
            freq="3h"
        )
        missing = full_index.difference(n.snapshots.get_level_values(1))
        print(f"Missing timesteps: {len(missing)}")
        print(missing)

        # Smooth if full-year data
        if len(price_series) > 1000:
            price_series = price_series.rolling(8, center=True).mean()

        # Plot
        axs[i].plot(
            price_series.index,
            price_series,
            color="darkred",
            linewidth=1
        )
        axs[i].set_ylabel("Market Price [$/MWh]")
        axs[i].set_xlabel("")
        axs[i].set_title(f"Load-weighted Market Price – {period}")
        axs[i].set_xlim(pd.Timestamp(f"{period}-01-01"), pd.Timestamp(f"{period+1}-01-01"))
        axs[i].set_ylim(0, 1190)

        # Major ticks at the 1st of each month
        axs[i].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Minor ticks at the 15th of each month (true midpoint)
        axs[i].xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))

        # Gridlines
        axs[i].grid(which="major", linestyle="-", linewidth=0.5, color="lightgrey", alpha=0.8)
        axs[i].grid(which="minor", linestyle="--", linewidth=0.3, color="lightgrey", alpha=0.6)

        # Rotate labels if needed
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()

    # Save figure
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved market price plot to {output_path}")


def plot_daily_curtailment(n: pypsa.Network, output_path: str, **wildcards) -> None:
    """
    Plot smoothed 3-hourly curtailment (MWh) for onwind, solar, and
    offwind_floating carriers. One subplot per investment period.
    Smoothed with an 8-step (24-hour) rolling mean for clarity.
    """

    CARRIERS = {
        "onwind":           ("Onshore Wind",           "#235ebc"),
        "solar":            ("Solar",                  "#f9d002"),
        "offwind_floating": ("Floating Offshore Wind", "#11a1c1"),
    }

    base_plot_size = 6
    num_periods = len(n.investment_periods)

    # ------------------------------------------------------------------
    # Compute curtailment per carrier across all investment periods
    # curtailment [MWh] = (p_max_pu * p_nom_opt - p) * snapshot_weightings
    # ------------------------------------------------------------------

    curtailment_by_carrier = {}

    for carrier, (label, color) in CARRIERS.items():

        gens = n.generators[n.generators.carrier == carrier].index

        if gens.empty:
            logger.warning(f"No generators found for carrier '{carrier}', skipping.")
            continue

        # Optimised capacity (MW) — fall back to p_nom for non-extendable
        p_nom = n.generators.loc[gens].apply(
            lambda row: row.p_nom_opt if row.p_nom_extendable else row.p_nom,
            axis=1,
        )

        # Available generation potential (MW)
        p_available = n.generators_t.p_max_pu[gens].multiply(p_nom, axis=1)

        # Actual dispatch (MW)
        p_dispatch = n.generators_t.p[gens]

        # Curtailment in MW, then convert to MWh using snapshot weightings
        weights = n.snapshot_weightings.generators  # accounts for 3h steps
        curtailment_mw = (p_available - p_dispatch).clip(lower=0)
        curtailment_mwh = curtailment_mw.multiply(weights, axis=0)

        # Sum across all generators of this carrier → single timeseries
        curtailment_by_carrier[carrier] = curtailment_mwh.sum(axis=1).mul(1e-3)  # MWh → GWh

        # # ------------------------------------------------------------------
        # # Sanity check: print total curtailment per carrier over full horizon
        # # ------------------------------------------------------------------

        # logger.info("=== Total curtailment sanity check ===")
        # for carrier, (label, color) in CARRIERS.items():
        #     if carrier not in curtailment_by_carrier:
        #         logger.info(f"  {label}: no data")
        #         continue
        #     # curtailment_by_carrier is already in GWh, convert back to MWh for the check
        #     total_mwh = curtailment_by_carrier[carrier].sum() * 1e3
        #     logger.info(f"  {label}: {total_mwh:,.1f} MWh")
        # logger.info("======================================")

    if not curtailment_by_carrier:
        logger.warning("No curtailment data found for any carrier. Skipping plot.")
        return

    # ------------------------------------------------------------------
    # Figure setup — one subplot per investment period
    # ------------------------------------------------------------------

    figsize = (14, base_plot_size * num_periods)
    fig, axs = plt.subplots(nrows=num_periods, ncols=1, figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, period in enumerate(n.investment_periods):

        for carrier, (label, color) in CARRIERS.items():

            if carrier not in curtailment_by_carrier:
                continue

            series = curtailment_by_carrier[carrier]

            # Select snapshots for this investment period
            sns = series.index[series.index.get_level_values(0) == period]
            carrier_series = series.loc[sns].droplevel("period")

            # Resample to daily totals (GWh/day)
            carrier_series = carrier_series.resample("D", origin = "start").sum()

            axs[i].plot(
                carrier_series.index,
                carrier_series,
                label=label,
                color=color,
                linewidth=1,
            )

        axs[i].set_ylabel("Curtailment [GWh/day]")
        axs[i].set_xlabel("")
        axs[i].set_title(f"Daily Curtailment by Technology – {period}")
        axs[i].legend(loc="upper right")
        axs[i].set_xlim(pd.Timestamp(f"{period}-01-01"), pd.Timestamp(f"{period+1}-01-01"))
        axs[i].set_ylim(0, 613)

    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved curtailment plot to {output_path}")


def plot_energy_production_area(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plot timeseries energy production (GWh) resampled to daily totals.

    One subplot per investment period, for the full year and each month separately.
    Energy is computed by multiplying power by snapshot weightings (handles 3h resolution).
    """

    # get data
    energy_mix = get_energy_timeseries(n)  # MW
    demand = get_demand_timeseries(n)       # MW

    # Convert MW → GWh using snapshot weightings (robust to any time resolution)
    weights = n.snapshot_weightings.generators  # hours per timestep
    energy_mix = energy_mix.multiply(weights, axis=0).mul(1e-3)  # MWh → GWh
    demand = demand.multiply(weights, axis=0).mul(1e-3)

    for carrier in energy_mix.columns:
        if "battery" in carrier or carrier in snakemake.params.electricity["extendable_carriers"]["StorageUnit"]:
            energy_mix[carrier + "_discharger"] = energy_mix[carrier].clip(lower=0.0001)
            energy_mix[carrier + "_charger"] = energy_mix[carrier].clip(upper=-0.0001)
            energy_mix = energy_mix.drop(columns=carrier)
            carriers_2_plot.append(f"{carrier}" + "_charger")
            carriers_2_plot.append(f"{carrier}" + "_discharger")
        if "H2" in carrier or carrier in snakemake.params.electricity["extendable_carriers"]["Store"]:
            energy_mix[carrier + "_discharger"] = energy_mix[carrier].clip(lower=0.0001)
            energy_mix[carrier + "_charger"] = energy_mix[carrier].clip(upper=-0.0001)
            energy_mix = energy_mix.drop(columns=carrier)
            carriers_2_plot.append(f"{carrier}" + "_charger")
            carriers_2_plot.append(f"{carrier}" + "_discharger")

    # Remove pure H2 carrier (energy store, not power)
    energy_mix = energy_mix[
        [col for col in energy_mix.columns if col != "H2"]
    ]

    carriers_2_plot = list(set(carriers_2_plot))
    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)

    color_palette = get_color_palette(n)

    months = n.snapshots.get_level_values(1).month.unique()
    num_periods = len(n.investment_periods)
    base_plot_size = 8

    FULL_WEEK_HOURS = 7 * 24

    for month in ["all", *months.to_list()]:
        figsize = (14, (base_plot_size * num_periods))
        fig, axs = plt.subplots(figsize=figsize, ncols=1, nrows=num_periods)

        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for i, investment_period in enumerate(n.investment_periods):

            if month == "all":
                sns = n.snapshots[
                    n.snapshots.get_level_values(0) == investment_period
                ]
            else:
                sns = n.snapshots[
                    (n.snapshots.get_level_values(0) == investment_period)
                    & (n.snapshots.get_level_values(1).month == month)
                ]

            # Extract and drop period level
            data = energy_mix.loc[sns].droplevel("period").round(2)
            dem = demand.loc[sns].droplevel("period").round(2)

            # ------------------------------------------------------------------
            # Resample to 7-day totals (fixed bins starting at period start)
            # ------------------------------------------------------------------
            data = data.resample("7D", origin="start").sum()
            dem = dem.resample("7D", origin="start").sum()

            # ------------------------------------------------------------------
            # FIX: Scale final partial week to full 7-day equivalent
            # ------------------------------------------------------------------

            # Get corresponding hours for these snapshots
            hours = weights.loc[sns].droplevel("period")
            weekly_hours = hours.resample("7D", origin="start").sum()

            scaling_factor = FULL_WEEK_HOURS / weekly_hours

            # Only scale bins that are shorter than a full week
            scaling_factor[weekly_hours >= FULL_WEEK_HOURS] = 1.0

            data = data.multiply(scaling_factor, axis=0)
            dem = dem.multiply(scaling_factor, axis=0)

            # print("Carriers being plotted (in stacking order):")
            # print(list(data.columns))

            positive_order = [
                "Nuclear",
                "Coal",
                "Oil",
                "Combined-Cycle Gas",
                "Open-Cycle Gas",
                "Ccgt-95Ccs", 
                "Reservoir & Dam",
                "Biomass",
                "Waste",
                "Geothermal",
                "battery_discharger",
                "4hr_battery_storage_discharger",
                "8hr_battery_storage_discharger",
                "H2 electrolysis_discharger",
                "100hr_battery_storage_discharger",
                "Solar",
                "Floating Offshore Wind",
                "Onshore Wind",
            ]

            negative_order = [
                "battery_charger",
                "4hr_battery_storage_charger",
                "8hr_battery_storage_charger",
                "H2 fuel cell_charger",
                "100hr_battery_storage_charger",
            ]

            # Keep only carriers that actually exist in the dataframe
            neg_existing = [c for c in negative_order if c in data.columns]
            pos_existing = [c for c in positive_order if c in data.columns]

            # Optional: catch anything you forgot
            remaining = [c for c in data.columns if c not in neg_existing + pos_existing]

            # Final stacking order
            data = data[neg_existing + pos_existing + remaining]

            # Plot
            data.plot.area(
                ax=axs[i],
                alpha=0.85,
                color=color_palette,
                linewidth=0,
            )

            dem.plot(
                ax=axs[i],
                ls="--",
                color="black",
                linewidth=3,
            )

            suffix = "-" + datetime.strptime(str(month), "%m").strftime("%b") if month != "all" else ""

            axs[i].legend(bbox_to_anchor=(1, 1), loc="upper left")
            axs[i].set_ylabel("Energy [GWh/week]")
            axs[i].set_xlabel("")
            axs[i].set_title(f"Weekly Energy Production – {investment_period}")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(create_title("Energy Production [GWh/week]", **wildcards))

        save = Path(save)
        fig.savefig(save.parent / (save.stem + suffix + save.suffix))
        plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_statistics",
            interconnect="texas",
            clusters=7,
            ll="v1.00",
            opts="REM-400SEG",
            sector="E",
        )
    configure_logging(snakemake)

    # extract shared plotting files
    n = pypsa.Network(snakemake.input.network)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)
    retirement_method = snakemake.params.retirement

    sanitize_carriers(n, snakemake.config)

    # mappers
    generating_link_carrier_map = {"fuel cell": "H2", "battery discharger": "battery"}

    # carriers to plot
    carriers = (
        snakemake.params.electricity["conventional_carriers"]
        + snakemake.params.electricity["renewable_carriers"]
        + snakemake.params.electricity["extendable_carriers"]["Generator"]
        + snakemake.params.electricity["extendable_carriers"]["StorageUnit"]
        + snakemake.params.electricity["extendable_carriers"]["Store"]
        + snakemake.params.electricity["extendable_carriers"]["Link"]
        + ["battery_charger", "battery_discharger", "imports"]
        + ["H2_charger", "H2_discharger"]
    )
    carriers = list(set(carriers))  # remove any duplicates

    # Export Statistics Tables
    groupers = n.statistics.groupers
    n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3).to_csv(
        snakemake.output.statistics_dissaggregated,
    )
    n.statistics().round(2).to_csv(snakemake.output.statistics_summary)
    n.generators.to_csv(snakemake.output.generators)
    n.storage_units.to_csv(snakemake.output.storage_units)
    # n.stores.to_csv(snakemake.output.stores)
    n.links.to_csv(snakemake.output.links)
    n.lines.to_csv(snakemake.output.lines)
    n.buses.to_csv(snakemake.output.buses)

    # Panel Plots
    plot_generator_data_panel(
        n,
        snakemake.output["generator_data_panel.pdf"],
        **snakemake.wildcards,
    )

    # Bar Plots
    plot_capacity_additions_bar(
        n,
        carriers,
        snakemake.output["capacity_additions_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_energy_capacity_additions_bar(
        n,
        carriers,
        snakemake.output.energy_capacity_additions_bar,
        **snakemake.wildcards,
    )
    plot_production_bar(
        n,
        carriers,
        snakemake.output["production_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_global_constraint_shadow_prices(
        n,
        snakemake.output["global_constraint_shadow_prices.pdf"],
        **snakemake.wildcards,
    )
    plot_regional_capacity_additions_bar(
        n,
        snakemake.output["bar_regional_capacity_additions.pdf"],
    )
    plot_regional_energy_capacity_additions_bar(
        n,
        snakemake.output.bar_regional_energy_capacity_additions,
    )
    plot_regional_production_bar(
        n,
        snakemake.output["bar_regional_production.pdf"],
    )
    plot_regional_energy_production_bar(
        n,
        snakemake.output.bar_regional_energy_production,
    )
    plot_regional_emissions_bar(
        n,
        snakemake.output["bar_regional_emissions.pdf"],
    )
    plot_emissions_bar(
        n,
        snakemake.output["bar_emissions.pdf"],
    )

    # Time Series Plots
    plot_production_area(
        n,
        carriers,
        snakemake.output["production_area.pdf"],
        **snakemake.wildcards,
    )
    plot_hourly_emissions(
        n,
        snakemake.output["emissions_area.pdf"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions_tech(
        n,
        snakemake.output["emissions_accumulated_tech.pdf"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions(
        n,
        snakemake.output["emissions_accumulated.pdf"],
        **snakemake.wildcards,
    )
    plot_fuel_costs(
        n,
        snakemake.output["fuel_costs.pdf"],
        **snakemake.wildcards,
    )

    # Box Plot
    plot_region_lmps(
        n,
        snakemake.output["region_lmps.pdf"],
        **snakemake.wildcards,
    )

    plot_storage_soc(
        n,
        snakemake.output["storage_soc.pdf"],
        **snakemake.wildcards,
    )

    plot_storage_soc_and_wind_cf(
        n,
        snakemake.output["storage_vs_wind.pdf"],
        **snakemake.wildcards,
    )

    # plot_storage_wind_complementarity_3(
    #     n,
    #     snakemake.output["storage_vs_wind_3.pdf"],
    #     **snakemake.wildcards,
    # )

    plot_market_price(
        n,
        snakemake.output["market_price.pdf"],
        **snakemake.wildcards,
    )

    plot_daily_curtailment(
        n,
        snakemake.output["daily_curtailment.pdf"],
        **snakemake.wildcards,
    )

    plot_energy_production_area(
        n,
        carriers,
        snakemake.output["energy_production_area.pdf"],
        **snakemake.wildcards,
    )
