"""Retirement risk: which cars were most likely to retire, and what the model leans on."""

import streamlit as st

from app import ui
from src import model

v = ui.view()
ui.header("Retirement risk", "A model trained on earlier seasons estimates each car's chance of not finishing, using "
                             "only what was known before the start. Pick a race to see who it worried about.")
ui.require(v.driver_race, "race entries")
rm = ui.get_reliability()
m = rm.metrics

tested = f"{min(rm.test_seasons)} and {max(rm.test_seasons)}" if len(rm.test_seasons) > 1 else str(rm.test_seasons[0])
ui.kpi_row([
    {"label": "How well it ranks risk", "value": f"{m['roc_auc']:.2f}",
     "delta": "0.5 is guessing, 1.0 is perfect", "delta_color": "off",
     "help": "ROC AUC on seasons the model never saw: the chance it rates a car that retired above one that finished."},
    {"label": "Cars that retired", "value": f"{m['base_rate_test']:.1%}", "delta": f"in {tested}", "delta_color": "off"},
    {"label": "Tested on", "value": tested, "delta": f"{m['n_test']:,} race starts", "delta_color": "off",
     "help": f"Trained on {m['n_train']:,} starts from the earlier seasons."},
])
ui.synthetic_banner("Whether a car retired is real. Most of the model's inputs (late parts, spares, spend) are "
                    "synthetic and were generated with a link to retirements, so the model finding them shows the "
                    "pipeline works; it is not a claim about why real cars fail.")

scored = rm.scored[rm.scored["season"].between(*v.seasons)]
if st.session_state.get("f_teams"):
    scored = scored[scored["team"].isin(st.session_state["f_teams"])]
if st.session_state.get("f_drivers"):
    scored = scored[scored["driver_id"].isin(st.session_state["f_drivers"])]

# ------------------------------------------------------------------ one race
ui.section("Risk for one race", "Cars sorted from highest to lowest predicted risk.")
ui.require(scored, "scored entries")
races = scored.drop_duplicates(["season", "round"]).sort_values(["season", "round"], ascending=False)
names = {f"{r.season} {r.race_name}": (r.season, r.round) for r in races.itertuples()}
had_dnf = scored.groupby(["season", "round"])["dnf"].any()
first_with_dnf = next((i for i, key in enumerate(names.values()) if had_dnf.get(key, False)), 0)
pick = st.selectbox("Race", list(names), index=first_with_dnf, key="rr_race",
                    help="Starts on the most recent race in which at least one car retired.")
season, rnd = names[pick]
board = scored[(scored["season"] == season) & (scored["round"] == rnd)].sort_values("risk", ascending=False).copy()
board["finished"] = ~board["dnf"].astype(bool)
st.dataframe(
    board[["driver_name", "constructor_name", "grid_clean", "risk", "risk_band", "finished", "status"]],
    hide_index=True,
    column_config={
        "driver_name": "Driver", "constructor_name": "Team",
        "grid_clean": st.column_config.NumberColumn("Started", format="P%d"),
        "risk": st.column_config.ProgressColumn("Predicted risk", min_value=0,
                                                max_value=float(max(0.5, scored["risk"].max())), format="percent"),
        "risk_band": "Level", "finished": st.column_config.CheckboxColumn("Finished"), "status": "Result",
    },
)
retired = board[~board["finished"]]
hit = int((~board.head(5)["finished"]).sum())
ui.takeaway(f"{len(retired)} of {len(board)} cars retired; {hit} of them were among the model's five highest-risk "
            f"cars. 'High' means at least 1.5 times the {m['base_rate_train']:.1%} retirement rate of the training seasons."
            + (" The model was trained on this season, so it has seen these results before."
               if (board["split"] == "train").all() else ""))

# ------------------------------------------------------------ what it uses
left, right = st.columns(2)
with left:
    ui.section("What the model leans on", "Longer bar = moves the risk score more.")
    ui.show(ui.memo("rr_importance", lambda: model.importance_chart(rm)), "rr_importance", table=True)
with right:
    ui.section("Risk by team", f"Average predicted risk against what actually happened, {tested}.")
    test_scored = scored[scored["split"] == "test (unseen)"]
    if test_scored.empty:
        st.info(f"Include {tested} in the season filter to see this chart.", icon=":material/info:")
    else:
        view_model = model.ReliabilityModel(**{**rm.__dict__, "scored": test_scored})
        ui.show(ui.memo("rr_team", lambda: model.risk_by_team(view_model)), "rr_team", table=True)

with st.expander("Model details: calibration, how each input pushes the score, more accuracy measures"):
    ui.show(ui.memo("rr_calibration", lambda: model.calibration_chart(rm)), "rr_calibration", table=True)
    ui.show(ui.memo("rr_shap", lambda: model.shap_summary(rm)), "rr_shap")
    feature = st.selectbox("Look at one input", rm.importance["feature"].tolist(), format_func=model.feature_label,
                           key="rr_feature")
    ui.show(ui.memo("rr_dependence", lambda: model.shap_dependence(rm, feature)), "rr_dependence")
    st.caption(f"Precision-recall AUC {m['pr_auc']:.2f} (guessing scores {m['base_rate_test']:.2f}). Brier score "
               f"{m['brier']:.4f} against {m['brier_baseline']:.4f} for always predicting the average rate; lower is "
               f"better. From 2023 the public results no longer give a cause for most retirements, so the model "
               f"predicts any retirement, mechanical or not.")
