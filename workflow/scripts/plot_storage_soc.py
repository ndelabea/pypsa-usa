import logging
import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from _helpers import configure_logging

logger = logging.getLogger(__name__)


def plot_storage_soc(n: pypsa.Network, output_path: str, **wildcards) -> None:
    """
    Plot monthly average state of charge (SOC) of extendable
    100hr battery storage units, aggregated by state using
    capacity-weighted averaging.
    """

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

    if "state" not in n.buses.columns:
        raise ValueError("No 'state' column found in buses.")

    su["state"] = su.bus.map(n.buses.state)
    su = su.dropna(subset=["state"])

    if su.empty:
        logger.warning("No storage units with valid state mapping.")
        return

    # -------------------------------------------------
    # Capacity-weighted SOC aggregation
    # -------------------------------------------------

    energy_capacity = su.p_nom_opt * su.max_hours
    soc = n.storage_units_t.state_of_charge[su.index]

    state_soc = {}

    for state, group in su.groupby("state"):
        units = group.index
        weights = energy_capacity.loc[units]

        weighted = soc[units].multiply(weights, axis=1)
        state_soc[state] = weighted.sum(axis=1) / weights.sum()

    state_soc = pd.DataFrame(state_soc)

    # Monthly averaging
    state_soc_monthly = state_soc.resample("M").mean()

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))
    state_soc_monthly.plot(ax=ax)

    ax.set_ylabel("Average State of Charge (MWh)")
    ax.set_title("Monthly Average SOC of Extendable 100hr Batteries by State")

    ax.legend(title="State", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    logger.info(f"Saved SOC plot to {output_path}")


# ==========================================================
# Main execution block (PyPSA-USA style)
# ==========================================================

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_storage_soc",
            interconnect="texas",
            clusters=7,
            ll="v1.00",
            opts="REM-400SEG",
            sector="E",
        )

    configure_logging(snakemake)

    logger.info("Loading network...")
    n = pypsa.Network(snakemake.input.network)

    plot_storage_soc(
        n,
        snakemake.output[0],
        **snakemake.wildcards,
    )