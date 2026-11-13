"""The daily BPM and RAPM leaderboard in the Shiny dashboard."""
import asyncio
from io import BytesIO
from urllib.request import Request, urlopen

import pandas as pd
from shiny import reactive, render, ui

CSV_URL = (
    "https://raw.githubusercontent.com/Siddhu-S/ncaa-player-dashboard/"
    "ratings-data/player-ratings.csv"
)
PLAYER_COLUMNS = {
    "player_name": "Player",
    "team": "Team",
    "bpm": "BPM",
    "rapm": "RAPM",
    "bpm_minutes": "BPM minutes",
    "rapm_possessions": "RAPM possessions",
}
COVERAGE_COLUMNS = {
    "season", "through_date", "bpm_through_date", "rapm_through_date", "expected_games",
    "bpm_games_included", "rapm_games_included", "rapm_excluded_possessions",
}


def ratings_tab_ui():
    return ui.div(
        {"id": "ratings-tab", "class": "tab-panel"},
        ui.div(
            {"class": "ratings-page"},
            ui.div(
                {"class": "ratings-heading"},
                ui.div(ui.h2("Player Ratings"), ui.output_ui("ratings_status")),
                ui.a("Open CSV", href=CSV_URL, target="_blank", rel="noopener",
                     class_="btn"),
            ),
            ui.div(
                {"class": "ratings-controls"},
                ui.input_text("ratings_search", "Player or team", placeholder="Search players or teams"),
                ui.input_select("ratings_sort", "Rank by", {"rapm": "RAPM", "bpm": "BPM"}),
            ),
            ui.div({"class": "ratings-grid"}, ui.output_data_frame("ratings_table")),
        ),
    )


async def fetch_csv():
    try:
        from pyodide.http import pyfetch
    except ModuleNotFoundError:
        def fetch():
            with urlopen(Request(CSV_URL, headers={"Cache-Control": "no-cache"}), timeout=15) as response:
                return response.read()

        return await asyncio.to_thread(fetch)

    response = await pyfetch(CSV_URL, cache="no-cache")
    if not response.ok:
        raise OSError(f"Ratings CSV returned HTTP {response.status}")
    return await response.bytes()


def read_ratings(raw):
    frame = pd.read_csv(BytesIO(raw), keep_default_na=False, na_values=[""])
    if frame.empty or not (set(PLAYER_COLUMNS) | COVERAGE_COLUMNS).issubset(frame.columns):
        raise ValueError("Ratings CSV is empty or has missing columns")
    for column in ("bpm", "rapm", "bpm_minutes", "rapm_possessions"):
        frame[column] = pd.to_numeric(frame[column])
    return frame


def ratings_tab_server(input):
    snapshot = reactive.Value(None)
    error = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.active_tab)
    async def refresh_ratings():
        if input.active_tab() != "ratings":
            return
        try:
            raw = await asyncio.wait_for(fetch_csv(), timeout=20)
            snapshot.set(read_ratings(raw))
            error.set(False)
        except (OSError, ValueError, RuntimeError):
            error.set(True)

    @render.ui
    def ratings_status():
        current = snapshot.get()
        if current is None:
            return ("The ratings CSV is unavailable. Reopen the tab to retry."
                    if error.get() else "Loading the daily ratings…")
        meta = current.iloc[0]
        season = int(meta["season"])
        expected = int(meta["expected_games"])
        return ui.TagList(
            ui.p(f"{season - 1}–{str(season)[-2:]} season · Games through {meta['through_date']}"),
            ui.p(
                f"BPM: {int(meta['bpm_games_included']):,} of {expected:,} games "
                f"through {meta['bpm_through_date']} · "
                f"RAPM: {int(meta['rapm_games_included']):,} of {expected:,} games "
                f"through {meta['rapm_through_date']} "
                f"({int(meta['rapm_excluded_possessions']):,} possessions excluded)"
            ),
            ui.p("Refresh failed. Showing the last loaded snapshot.") if error.get() else None,
        )

    @render.data_frame
    def ratings_table():
        frame = snapshot.get()
        if frame is None:
            return render.DataGrid(pd.DataFrame(columns=list(PLAYER_COLUMNS.values())), summary=False)

        search = input.ratings_search().strip()
        if search:
            matches = frame["player_name"].str.contains(search, case=False, regex=False, na=False)
            matches |= frame["team"].str.contains(search, case=False, regex=False, na=False)
            frame = frame.loc[matches]
        order = input.ratings_sort()
        frame = frame.sort_values([order, "player_name", "team"], ascending=[False, True, True])
        display = frame[list(PLAYER_COLUMNS)].round({"bpm": 2, "rapm": 2}).rename(columns=PLAYER_COLUMNS)
        return render.DataGrid(display, height="min(58vh, 670px)", width="100%", summary=False)
