"""
team_manager_page.py — Streamlit page for managing team name aliases.

This page shows:
1. Stats (internal teams, custom aliases, new teams added, unmatched count)
2. Unmatched teams from all sources, grouped by suggested matches
3. UI to:
   - Group unmatched teams that are the same team
   - Choose a canonical name
   - Add the team to the model (with default ratings if new)
   - Add aliases to the config file

Integrate into streamlit_app.py with:
    from team_manager_page import render_team_manager_page
    render_team_manager_page()
"""
import sys
import os
import json
from pathlib import Path
from collections import defaultdict

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from team_alias_manager import (
    load_custom_aliases, save_custom_aliases,
    load_new_teams, save_new_teams,
    load_unmatched_teams, get_all_unmatched_names,
    suggest_team_groupings, get_team_manager_stats,
    add_new_team_to_model, get_internal_teams,
)


def render_team_manager_page():
    """Render the Team Manager page in Streamlit."""
    st.header("🏒 Team Manager")
    st.caption("Match unmatched team names across data sources and add new teams to the model")

    # === STATS ===
    stats = get_team_manager_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Internal Teams", stats["internal_teams"])
    c2.metric("Custom Aliases", stats["custom_aliases_total"])
    c3.metric("New Teams Added", stats["new_teams_added"])
    c4.metric("Unmatched (last audit)", stats["unmatched_teams"])

    st.divider()

    # === RUN AUDIT BUTTON ===
    st.subheader("Step 1: Detect Unmatched Teams")
    st.caption("Run the team name audit to find teams that don't match across sources")
    if st.button("🔍 Run Team Name Audit", help="Fetches team names from all 6 sources and checks normalization. Takes ~2 min."):
        with st.spinner("Running audit..."):
            import subprocess
            proc = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "team_name_audit.py")],
                cwd=str(PROJECT_ROOT), timeout=600,
                capture_output=True, text=True
            )
            if proc.returncode == 0:
                st.success("✅ Audit complete")
            else:
                st.error(f"❌ Failed: {proc.stderr[-500:]}")
        st.rerun()

    # === SHOW UNMATCHED TEAMS ===
    st.divider()
    st.subheader("Step 2: Match Unmatched Teams")
    unmatched = get_all_unmatched_names()
    if not unmatched:
        st.success("✅ No unmatched teams! All team names normalize correctly.")
        return

    st.caption(f"Found {len(unmatched)} unique unmatched team names across all sources")

    # Show suggested groupings
    groups = suggest_team_groupings()
    
    st.write("**Suggested groupings** (teams with similar names that might be the same):")
    st.caption("Review each group. If teams ARE the same, confirm the canonical name and click 'Add to Model'.")

    # Show each group with an expander
    for i, group in enumerate(groups):
        with st.expander(
            f"{'🔗 GROUP' if not group['single'] else '❓ SINGLE'}: {group['suggested_name']} "
            f"({len(group['members'])} name{'s' if len(group['members']) > 1 else ''})",
            expanded=not group["single"]  # expand groups, collapse singles
        ):
            # Show all member names with their sources
            for member in group["members"]:
                st.write(f"  • **{member}**")
            
            # Show which sources have these names
            st.write("**Sources:**")
            for source_info in group["sources"]:
                st.write(f"  - {source_info['source']} / {source_info['league']}")

            # Let user choose canonical name
            st.write("**Choose canonical name:**")
            options = group["members"] + ["(type custom name)"]
            canonical_choice = st.radio(
                f"Canonical name for group {i+1}",
                options,
                index=0,
                label_visibility="collapsed",
                key=f"canonical_{i}"
            )

            if canonical_choice == "(type custom name)":
                canonical_choice = st.text_input(
                    "Enter custom canonical name:",
                    value=group["suggested_name"],
                    key=f"custom_name_{i}"
                )

            # Check if canonical name exists in model
            internal_teams = get_internal_teams()
            exists_in_model = canonical_choice in internal_teams

            if exists_in_model:
                st.success(f"✅ '{canonical_choice}' already exists in the model — aliases will be added")
            else:
                st.warning(f"⚠️ '{canonical_choice}' does NOT exist in the model. "
                          f"You can add it with default ratings (league average).")

            # Buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"➕ Add Aliases", key=f"add_aliases_{i}",
                             help="Save the alias mappings (canonical name → all variants)"):
                    # Save aliases
                    aliases = load_custom_aliases()
                    if canonical_choice not in aliases:
                        aliases[canonical_choice] = []
                    for member in group["members"]:
                        if member.lower() != canonical_choice.lower():
                            if member.lower() not in aliases[canonical_choice]:
                                aliases[canonical_choice].append(member.lower())
                    save_custom_aliases(aliases)
                    st.success(f"✅ Added {len(group['members']) - 1} aliases for '{canonical_choice}'")
                    st.rerun()

            with col2:
                if not exists_in_model:
                    if st.button(f"🆕 Add Team (with Elo)", key=f"add_team_{i}",
                                 help="Fetch Elo from beyondELO, initialize DC state, and add aliases. "
                                      "If beyondELO has recent data → use it. "
                                      "If old/no data → use promoted-team default (Elo=1400)."):
                        # Determine league from sources
                        league = group["sources"][0]["league"] if group["sources"] else None
                        
                        with st.spinner(f"Fetching Elo from beyondELO for '{canonical_choice}'..."):
                            from team_alias_manager import add_new_team_to_model
                            result = add_new_team_to_model(canonical_choice, league, use_elo=True)
                        
                        if result["success"]:
                            st.success(f"✅ Added '{canonical_choice}' to model")
                            st.info(f"**Elo source:** {result['elo_source']}\n"
                                    f"**Initial Elo:** {result['initial_elo']:.0f}\n"
                                    f"**Attack:** {result['attack']:+.4f}\n"
                                    f"**Defence:** {result['defence']:+.4f}\n"
                                    f"**Match history:** {result['match_history_fetched']} matches")
                            
                            # Also add aliases
                            aliases = load_custom_aliases()
                            if canonical_choice not in aliases:
                                aliases[canonical_choice] = []
                            for member in group["members"]:
                                if member.lower() != canonical_choice.lower():
                                    if member.lower() not in aliases[canonical_choice]:
                                        aliases[canonical_choice].append(member.lower())
                            save_custom_aliases(aliases)
                        else:
                            st.warning(f"Team '{canonical_choice}' already exists in model")
                        st.rerun()

    # === MANUAL ALIAS ADDITION ===
    st.divider()
    st.subheader("Step 3: Add Alias Manually")
    st.caption("If you know a team name mapping that's not shown above, add it here")
    
    with st.form("manual_alias_form"):
        col1, col2 = st.columns(2)
        with col1:
            manual_canonical = st.text_input("Canonical (internal) name", 
                                              help="The team name in our model, e.g., 'Man United'")
        with col2:
            manual_alias = st.text_input("Alias (external name)",
                                         help="The name used by an external source, e.g., 'Manchester United'")
        
        if st.form_submit_button("Add Alias"):
            if manual_canonical and manual_alias:
                aliases = load_custom_aliases()
                if manual_canonical not in aliases:
                    aliases[manual_canonical] = []
                if manual_alias.lower() not in aliases[manual_canonical]:
                    aliases[manual_canonical].append(manual_alias.lower())
                    save_custom_aliases(aliases)
                    st.success(f"✅ Added alias: '{manual_alias}' → '{manual_canonical}'")
                else:
                    st.info("Alias already exists")
            else:
                st.error("Please fill in both fields")
        st.rerun()

    # === SHOW CURRENT ALIASES ===
    st.divider()
    st.subheader("Current Custom Aliases")
    aliases = load_custom_aliases()
    if aliases:
        for canonical, alias_list in sorted(aliases.items()):
            with st.expander(f"{canonical} ({len(alias_list)} aliases)"):
                for alias in alias_list:
                    st.write(f"  • {alias}")
                    if st.button(f"Remove '{alias}'", key=f"rm_{canonical}_{alias}"):
                        aliases[canonical].remove(alias)
                        if not aliases[canonical]:
                            del aliases[canonical]
                        save_custom_aliases(aliases)
                        st.rerun()
    else:
        st.info("No custom aliases yet. Run the audit and match teams above.")

    # === SHOW NEW TEAMS ===
    st.divider()
    st.subheader("Newly Added Teams (Elo-based initialization)")
    new_teams = load_new_teams()
    if new_teams:
        for team_name, info in new_teams.items():
            with st.expander(f"{team_name} — Elo={info.get('initial_elo', '?'):.0f} "
                           f"({info.get('elo_source', '?')[:60]})"):
                st.write(f"**League:** {info.get('league', '?')}")
                st.write(f"**Initial Elo:** {info.get('initial_elo', '?'):.0f}")
                st.write(f"**Elo source:** {info.get('elo_source', '?')}")
                st.write(f"**DC Attack:** {info.get('default_attack', 0):+.4f}")
                st.write(f"**DC Defence:** {info.get('default_defence', 0):+.4f}")
                st.write(f"**beyondELO name:** {info.get('beyondelo_name', 'N/A')}")
                st.write(f"**Match history fetched:** {info.get('match_history_fetched', 0)} matches")
                st.write(f"**Added:** {info.get('added_date', '?')[:10]}")
    else:
        st.info("No new teams added yet.")

    # === REFRESH MODEL ===
    st.divider()
    st.subheader("Step 4: Apply Changes")
    st.caption("After adding aliases/teams, reload the model to apply changes")
    if st.button("🔄 Reload Model", help="Clear the cached model so it reloads with new aliases"):
        try:
            st.cache_resource.clear()
            st.success("✅ Model cache cleared. The app will reload with new aliases on next interaction.")
        except:
            st.info("Cache cleared")
