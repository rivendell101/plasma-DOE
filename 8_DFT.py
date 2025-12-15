import streamlit as st
import pandas as pd
import numpy as np
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from itertools import product, combinations
import seaborn as sns
from PIL import Image
from matplotlib import cm 
from scipy.optimize import minimize
import plotly.graph_objects as go
import itertools
from io import BytesIO

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
        }
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="DOE & ANOVA App", layout="wide")

st.sidebar.title("Design of Experiments")
st.sidebar.markdown("----")

selected_page = st.sidebar.radio("Navigation Menu", ["Screening Design(4+)", "Factorial Design (2-3)",
     "Response Surface Methodology (2-3)", "Linear Regression (1)"])
set_custom_style()

if selected_page == "Screening Design(4+)":
    st.title("Factorial Design Application (Res V)")
    st.markdown(
        "This application generates a 4+ factor screening design (2-level DOE) "
        "with randomized runs and empty response columns."
    )

    import pandas as pd
    import itertools
    from io import BytesIO

    st.subheader("⚙️ Define Factors for DOE")

    # Ask user how many factors
    num_factors = st.number_input(
        "How many factors do you want to include?",
        min_value=1, max_value=10, value=4
    )

    # Store factor information
    factors = {}

    with st.expander("🧪 Factor Inputs", expanded=True):
        for i in range(num_factors):
            st.markdown(f"### Factor {i+1}")
            col1, col2, col3 = st.columns(3)

            with col1:
                factor_name = st.text_input(f"Name of Factor {i+1}", key=f"name_{i}")
            with col2:
                min_val = st.number_input(f"Minimum value", key=f"min_{i}")
            with col3:
                max_val = st.number_input(f"Maximum value", key=f"max_{i}")

            # only store if it has a name
            if factor_name:
                factors[factor_name] = (min_val, max_val)

    # --- RESPONSE SECTION ---
    st.subheader("📈 Define Response Variables (Outputs)")
    num_responses = st.number_input(
        "How many response variables do you need?",
        min_value=1, max_value=10, value=1
    )

    responses = []
    with st.expander("📌 Response Inputs", expanded=True):
        for i in range(num_responses):
            response_name = st.text_input(f"Name of Response {i+1}", key=f"response_{i}")
            if response_name:
                responses.append(response_name)

    # ===== APPLY BUTTON → DOE WITH FACTORS + RESPONSES =====
    if st.button("⚡ Apply & Generate Design of Experiments"):

        if len(factors) >= 4:
            st.success("Generating **Screening Design (Full Factorial, Randomized)**")

            # 1) build full factorial based on factor min/max
            factor_names = list(factors.keys())  # preserves input order
            factor_levels = [[factors[name][0], factors[name][1]] for name in factor_names]

            runs = list(itertools.product(*factor_levels))

            # Base DOE: only factor columns
            doe_df = pd.DataFrame(runs, columns=factor_names)

            # Randomize order and add Run column
            doe_df = doe_df.sample(frac=1).reset_index(drop=True)
            doe_df.insert(0, "Run", range(1, len(doe_df) + 1))

            # 2) Add response columns at the end (empty)
            for r in responses:
                doe_df[r] = None

            # 3) Enforce column order: Run, factors…, responses…
            ordered_cols = ["Run"] + factor_names + responses
            doe_df = doe_df[ordered_cols]

            # ── THIS is the final DOE (also the downloadable one) ──
            st.subheader("📊 Design of Experiments (Factors + Responses)")
            st.dataframe(doe_df, hide_index=True)

            # --- DOWNLOAD DOE AS CSV (WITH RESPONSES) ---
            csv_data = doe_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download DOE (CSV)",
                csv_data,
                "screening_design.csv",
                mime="text/csv"
            )

               # --- DOWNLOAD DOE AS EXCEL (Using openpyxl instead of xlsxwriter) ---
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            doe_df.to_excel(writer, index=False, sheet_name="DOE")

        st.download_button(
            "📥 Download DOE (Excel)",
            output.getvalue(),
            "screening_design.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.warning("⚠️ Screening design requires **at least 4 named factors**.")

        # ========================================================================
    #  UPLOAD COMPLETED DOE FILE & RUN ANOVA SECTION
    # ========================================================================

    st.markdown("---")
    st.subheader("📤 Upload Completed DOE (with Responses) to Run ANOVA")

    uploaded_file = st.file_uploader("Upload your DOE file (.csv or .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the file
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.success("File successfully uploaded!")
            st.dataframe(data.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")

        # Ensure responses exist
    if not responses or not all(r in data.columns for r in responses):
        st.warning("⚠ Your data must contain **response columns** to perform ANOVA.")
    else:
        # Select which response to analyze
        selected_response = st.selectbox("Choose Response for ANOVA", responses)

        # AUTO-DETECT FACTOR NAMES FROM UPLOADED DOE
        factor_names = [col for col in data.columns if col not in ["Run"] + responses]

        # Build formula dynamically
        formula = f"{selected_response} ~ " + " + ".join(factor_names)
        st.code(f"Model Formula:  {formula}", language="python")

        try:
            model = ols(formula, data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            st.subheader("📊 ANOVA Table")
            st.write(anova_table)

            st.subheader("📑 Model Summary")
            st.text(model.summary())

            # Residual diagnostics
            residuals = pd.DataFrame({
                "Fitted": model.fittedvalues,
                "Residuals": model.resid
            })
            st.subheader("📉 Residuals & Error Table")
            st.dataframe(residuals)

            # Plot diagnostics
            st.subheader("📈 ANOVA Diagnostics")

            fig1, ax1 = plt.subplots()
            ax1.scatter(model.fittedvalues, model.resid)
            ax1.axhline(0, linestyle="--")
            ax1.set_xlabel("Fitted Values")
            ax1.set_ylabel("Residuals")
            ax1.set_title("Residuals vs Fitted")
            st.pyplot(fig1)

            fig2 = sm.qqplot(model.resid, line='45', fit=True)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ ANOVA failed: {e}")
# ==============================================================
# 2–3 FACTORS → FULL FACTORIAL ONLY
# ==============================================================
elif selected_page == "Factorial Design (2-3)":
    st.title("Factorial Design (2–3 Factors)")
    st.markdown("This page generates a **2-level full factorial DOE** with 2–3 factors, plus ANOVA and plots.")

    # ---------- DEFINE FACTORS ----------
    st.subheader("⚙️ Define Factors")
    num_factors = st.number_input("How many factors? (Must be 2–3)", min_value=2, max_value=3, value=2, key="ff_num_factors")

    factors = {}
    with st.expander("🧪 Factor Input", expanded=True):
        for i in range(num_factors):
            st.markdown(f"### Factor {i+1}")
            col1, col2, col3 = st.columns(3)
            with col1:
                factor_name = st.text_input(f"Name of Factor {i+1}", key=f"ff_f_name_{i}")
            with col2:
                min_val = st.number_input(f"Min value", key=f"ff_f_min_{i}")
            with col3:
                max_val = st.number_input(f"Max value", key=f"ff_f_max_{i}")
            if factor_name:
                factors[factor_name] = (min_val, max_val)

    # ---------- DEFINE RESPONSES ----------
    st.subheader("📈 Response Variables")
    num_responses = st.number_input("Number of Responses", min_value=1, max_value=10, value=1, key="ff_num_responses")
    responses = []
    with st.expander("📌 Response Inputs", expanded=True):
        for i in range(num_responses):
            rname = st.text_input(f"Response {i+1} Name", key=f"ff_resp_{i}")
            if rname:
                responses.append(rname)

    # ---------- GENERATE FULL FACTORIAL DOE ----------
    doe_df = None

    if st.button("⚡ Generate Full Factorial DOE", key="ff_generate_button"):
        factor_names = list(factors.keys())
        levels = [[factors[name][0], factors[name][1]] for name in factor_names]

        if len(factor_names) not in [2, 3]:
            st.error("Only **2 or 3 factors** are allowed!")
        else:
            st.success("✨ FULL FACTORIAL DOE GENERATED")

            runs = list(itertools.product(*levels))
            doe_df = pd.DataFrame(runs, columns=factor_names)

            # Randomize & add responses
            doe_df = doe_df.sample(frac=1).reset_index(drop=True)
            doe_df.insert(0, "Run", range(1, len(doe_df) + 1))
            for r in responses:
                doe_df[r] = None

            st.subheader("📊 Full Factorial DOE (with Responses)")
            st.dataframe(doe_df, hide_index=True)

            # Download
            csv_data = doe_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download DOE (CSV)", csv_data, "doe_full_factorial_2_3.csv", mime="text/csv", key="ff_csv")

            out = BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                doe_df.to_excel(writer, index=False, sheet_name="DOE")
            st.download_button(
                "📥 Download DOE (Excel)",
                out.getvalue(),
                "doe_full_factorial_2_3.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="ff_xlsx"
            )

    # ---------- UPLOAD FOR ANOVA ----------
    st.markdown("---")
    st.subheader("📤 Upload Completed DOE (with Responses) for ANOVA")

    uploaded_file_ff = st.file_uploader("Upload DOE (.csv or .xlsx)", type=["csv", "xlsx"], key="ff_upload")

    data_ff = None
    if uploaded_file_ff is not None:
        try:
            if uploaded_file_ff.name.endswith(".csv"):
                data_ff = pd.read_csv(uploaded_file_ff)
            else:
                data_ff = pd.read_excel(uploaded_file_ff)

            st.success("File loaded successfully!")
            st.dataframe(data_ff.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            data_ff = None

    if data_ff is not None:
        # Ensure response columns exist
        if not responses or not all(r in data_ff.columns for r in responses):
            st.warning("⚠ Your data must contain the **response columns** you defined above.")
        else:
            selected_response_ff = st.selectbox("Select Response for ANOVA:", responses, key="ff_resp_for_anova")

            # factor names from uploaded data
            ff_factor_names = [col for col in data_ff.columns if col not in ["Run"] + responses]

            # build model: main effects + 2-way interactions
            formula_ff = f"{selected_response_ff} ~ " + " + ".join(ff_factor_names)
            if len(ff_factor_names) > 1:
                formula_ff += " + " + " + ".join([f"{a}:{b}" for a, b in itertools.combinations(ff_factor_names, 2)])

            st.code(f"Model Formula:\n{formula_ff}", language="python")

            try:
                # drop rows with NaNs in relevant columns to avoid inf/NaN error
                clean_ff = data_ff.dropna(subset=[selected_response_ff] + ff_factor_names)
                if clean_ff.empty:
                    st.error("❌ No complete rows (no NaNs) for this response and factors. Fill your DOE first.")
                else:
                    model_ff = ols(formula_ff, data=clean_ff).fit()
                    anova_ff = sm.stats.anova_lm(model_ff, typ=2)

                    st.subheader("📊 ANOVA Table")
                    st.write(anova_ff)

                    st.subheader("📑 Model Summary")
                    st.text(model_ff.summary())

                    # Residuals
                    residuals_ff = pd.DataFrame({
                        "Fitted": model_ff.fittedvalues,
                        "Residuals": model_ff.resid
                    })
                    st.subheader("📉 Residual Diagnostics")
                    st.write("🔎 Residual Table")
                    st.dataframe(residuals_ff)

                    # Residuals vs Fitted
                    fig1, ax1 = plt.subplots()
                    ax1.scatter(model_ff.fittedvalues, model_ff.resid)
                    ax1.axhline(0, linestyle="--")
                    ax1.set_xlabel("Fitted Values")
                    ax1.set_ylabel("Residuals")
                    ax1.set_title("Residuals vs Fitted")
                    st.pyplot(fig1)

                    # QQ Plot
                    fig2 = sm.qqplot(model_ff.resid, line="45", fit=True)
                    st.pyplot(fig2)

                    # ---------- EXTRA: FACTOR vs RESPONSE PLOT ----------
                    st.markdown("---")
                    st.subheader("📈 Plot Factor vs Response")

                    x_factor_ff = st.selectbox(
                        "Select Factor (X-axis)",
                        ff_factor_names,
                        key="ff_plot_factor"
                    )
                    y_resp_plot_ff = st.selectbox(
                        "Select Response (Y-axis)",
                        responses,
                        index=responses.index(selected_response_ff) if selected_response_ff in responses else 0,
                        key="ff_plot_resp"
                    )

                    plot_df_ff = clean_ff[[x_factor_ff, y_resp_plot_ff]].dropna()
                    plot_df_ff = plot_df_ff.sort_values(by=x_factor_ff)

                    fig3, ax3 = plt.subplots()
                    ax3.plot(plot_df_ff[x_factor_ff], plot_df_ff[y_resp_plot_ff], marker="o")
                    ax3.set_xlabel(x_factor_ff)
                    ax3.set_ylabel(y_resp_plot_ff)
                    ax3.set_title(f"{y_resp_plot_ff} vs {x_factor_ff}")
                    st.pyplot(fig3)

            except Exception as e:
                st.error(f"❌ ANOVA failed: {e}")   


# ==============================================================
# 2–3 FACTORS → RESPONSE SURFACE (CCD) ONLY
# ==============================================================
elif selected_page == "Response Surface Methodology (2-3)":
    st.title("Response Surface Methodology (2–3 Factors)")
    st.markdown("This page generates a **Central Composite Design (CCD)** and performs RSM ANOVA plus contour plots.")

    # ---------- DEFINE FACTORS ----------
    st.subheader("⚙️ Define Factors")
    num_factors = st.number_input("How many factors? (Must be 2–3)", min_value=2, max_value=3, value=2, key="rsm_num_factors")

    factors = {}
    with st.expander("🧪 Factor Input", expanded=True):
        for i in range(num_factors):
            st.markdown(f"### Factor {i+1}")
            col1, col2, col3 = st.columns(3)
            with col1:
                factor_name = st.text_input(f"Name of Factor {i+1}", key=f"rsm_f_name_{i}")
            with col2:
                min_val = st.number_input(f"Min value", key=f"rsm_f_min_{i}")
            with col3:
                max_val = st.number_input(f"Max value", key=f"rsm_f_max_{i}")
            if factor_name:
                factors[factor_name] = (min_val, max_val)

    # ---------- DEFINE RESPONSES ----------
    st.subheader("📈 Response Variables")
    num_responses = st.number_input("Number of Responses", min_value=1, max_value=10, value=1, key="rsm_num_responses2")
    responses = []
    with st.expander("📌 Response Inputs", expanded=True):
        for i in range(num_responses):
            rname = st.text_input(f"Response {i+1} Name", key=f"rsm_resp2_{i}")
            if rname:
                responses.append(rname)

    # ---------- GENERATE CCD DOE ----------
    doe_rsm = None

    if st.button("⚡ Generate CCD DOE", key="rsm_generate_button"):
        factor_names = list(factors.keys())

        if len(factor_names) not in [2, 3]:
            st.error("Only **2 or 3 factors** are allowed!")
        else:
            st.success("✨ CCD (RSM) DOE GENERATED")

            k = len(factor_names)
            alpha = 1.414  # rotatable CCD

            factorial = np.array(list(itertools.product([-1, 1], repeat=k)))

            axial = []
            for i in range(k):
                arr = np.zeros(k)
                arr[i] = alpha
                axial.append(arr.copy())
                arr[i] = -alpha
                axial.append(arr.copy())
            center = np.zeros((3, k))  # 3 center points

            X_coded = np.vstack([factorial, axial, center])

            # Convert coded → real
            real = {}
            for j, name in enumerate(factor_names):
                low, high = factors[name]
                center_pt = (low + high) / 2
                half_range = (high - low) / 2
                real[name] = center_pt + X_coded[:, j] * half_range

            doe_rsm = pd.DataFrame(real)

            doe_rsm = doe_rsm.sample(frac=1).reset_index(drop=True)
            doe_rsm.insert(0, "Run", range(1, len(doe_rsm) + 1))
            for r in responses:
                doe_rsm[r] = None

            st.subheader("📊 CCD DOE (with Responses)")
            st.dataframe(doe_rsm, hide_index=True)

            # Download
            csv_rsm = doe_rsm.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download DOE (CSV)", csv_rsm, "doe_rsm_2_3.csv", mime="text/csv", key="rsm_csv")

            out_rsm = BytesIO()
            with pd.ExcelWriter(out_rsm, engine="openpyxl") as writer:
                doe_rsm.to_excel(writer, index=False, sheet_name="DOE")
            st.download_button(
                "📥 Download DOE (Excel)",
                out_rsm.getvalue(),
                "doe_rsm_2_3.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="rsm_xlsx"
            )

    # ---------- UPLOAD FOR ANOVA ----------
    st.markdown("---")
    st.subheader("📤 Upload Completed CCD DOE (with Responses) for ANOVA")

    uploaded_file_rsm = st.file_uploader("Upload DOE (.csv or .xlsx)", type=["csv", "xlsx"], key="rsm_upload2")

    data_rsm = None
    if uploaded_file_rsm is not None:
        try:
            if uploaded_file_rsm.name.endswith(".csv"):
                data_rsm = pd.read_csv(uploaded_file_rsm)
            else:
                data_rsm = pd.read_excel(uploaded_file_rsm)

            st.success("File loaded successfully!")
            st.dataframe(data_rsm.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            data_rsm = None

    if data_rsm is not None:
        if not responses or not all(r in data_rsm.columns for r in responses):
            st.warning("⚠ Your data must contain the **response columns** you defined above.")
        else:
            selected_response_rsm = st.selectbox("Select Response for ANOVA:", responses, key="rsm_resp_for_anova2")

            rsm_factor_names = [col for col in data_rsm.columns if col not in ["Run"] + responses]

            main_terms = " + ".join(rsm_factor_names)
            quad_terms = " + ".join([f"np.power({f}, 2)" for f in rsm_factor_names])
            if len(rsm_factor_names) > 1:
                inter_terms = " + ".join([f"{a}:{b}" for a, b in itertools.combinations(rsm_factor_names, 2)])
                formula_rsm = f"{selected_response_rsm} ~ {main_terms} + {quad_terms} + {inter_terms}"
            else:
                formula_rsm = f"{selected_response_rsm} ~ {main_terms} + {quad_terms}"

            st.code(f"Model Formula:\n{formula_rsm}", language="python")

            try:
                # Drop rows with NaNs for safety
                clean_rsm = data_rsm.dropna(subset=[selected_response_rsm] + rsm_factor_names)
                if clean_rsm.empty:
                    st.error("❌ No complete rows (no NaNs) for this response and factors. Fill your DOE first.")
                else:
                    model_rsm = ols(formula_rsm, data=clean_rsm).fit()
                    anova_rsm = sm.stats.anova_lm(model_rsm, typ=2)

                    st.subheader("📊 ANOVA Table")
                    st.write(anova_rsm)

                    st.subheader("📑 Model Summary")
                    st.text(model_rsm.summary())

                    # Residuals
                    residuals_rsm = pd.DataFrame({
                        "Fitted": model_rsm.fittedvalues,
                        "Residuals": model_rsm.resid
                    })
                    st.subheader("📉 Residual Diagnostics")
                    st.write("🔎 Residual Table")
                    st.dataframe(residuals_rsm)

                    # Residuals vs Fitted
                    fig1, ax1 = plt.subplots()
                    ax1.scatter(model_rsm.fittedvalues, model_rsm.resid)
                    ax1.axhline(0, linestyle="--")
                    ax1.set_xlabel("Fitted Values")
                    ax1.set_ylabel("Residuals")
                    ax1.set_title("Residuals vs Fitted")
                    st.pyplot(fig1)

                    # QQ Plot
                    fig2 = sm.qqplot(model_rsm.resid, line="45", fit=True)
                    st.pyplot(fig2)

                    # ---------- CONTOUR PLOTS ----------
                    st.markdown("---")
                    st.subheader("🌈 Contour Plot: Factors vs Response")

                    rsm_factor_names = [col for col in data_rsm.columns if col not in ["Run"] + responses]

                    y_resp_plot_rsm = st.selectbox(
                        "Select Response for Contour Plot",
                        responses,
                        index=responses.index(selected_response_rsm) if selected_response_rsm in responses else 0,
                        key="rsm_vis_resp2"
                    )

                    selected_factors_rsm = st.multiselect(
                        "Select TWO factors for Contour Plot (X & Y):",
                        rsm_factor_names,
                        max_selections=2,
                        key="rsm_vis_factors2"
                    )

                    if len(selected_factors_rsm) == 2:
                        x_factor, y_factor = selected_factors_rsm

                        try:
                            pivot = clean_rsm.pivot_table(
                                index=x_factor,
                                columns=y_factor,
                                values=y_resp_plot_rsm,
                                aggfunc="mean"
                            ).sort_index().sort_index(axis=1)

                            X = pivot.index.values
                            Y = pivot.columns.values
                            Z = pivot.values

                            if Z.size == 0:
                                st.error("No data available to build contour (check your DOE table).")
                            else:
                                fig_ct = go.Figure(
                                    data=[
                                        go.Contour(
                                            x=X,
                                            y=Y,
                                            z=Z,
                                            contours=dict(showlabels=True)
                                        )
                                    ]
                                )
                                fig_ct.update_layout(
                                    title=f"Contour Plot — {y_resp_plot_rsm}",
                                    xaxis_title=x_factor,
                                    yaxis_title=y_factor,
                                )
                                st.plotly_chart(fig_ct, use_container_width=True)

                        except Exception as e:
                            st.error(f"⚠ Could not create contour plot: {e}")
                    else:
                        st.info("📌 Please select **exactly TWO factors** for the contour plot.")

            except Exception as e:
                st.error(f"❌ ANOVA failed: {e}")


elif selected_page == "Linear Regression (1)":
    st.title("Linear Regression (1 Factor)")
    st.markdown(
        "This page creates a **1-factor design of experiments (DOE)** from a factor range, "
        "then lets you upload the completed DOE to run **linear regression, ANOVA, and diagnostics**."
    )

    # --------------------------------------------------------------
    # 1) DEFINE FACTOR + RESPONSES & GENERATE 1-FACTOR DOE
    # --------------------------------------------------------------
    st.subheader("⚙️ Define Factor for DOE")

    lr_factor_name = st.text_input("Name of Factor (X)", key="lr_factor_name")
    col1, col2, col3 = st.columns(3)
    with col1:
        lr_min = st.number_input("Minimum value", key="lr_min")
    with col2:
        lr_max = st.number_input("Maximum value", key="lr_max")
    with col3:
        lr_levels = st.number_input(
            "Number of levels / runs",
            min_value=2, max_value=50, value=5, step=1,
            key="lr_levels"
        )

    st.subheader("📈 Define Response Variables (Y)")
    lr_num_responses = st.number_input(
        "How many response variables do you need?",
        min_value=1, max_value=10, value=1,
        key="lr_num_responses"
    )

    lr_responses = []
    with st.expander("📌 Response Inputs", expanded=True):
        for i in range(lr_num_responses):
            rname = st.text_input(f"Name of Response {i+1}", key=f"lr_resp_{i}")
            if rname:
                lr_responses.append(rname)

    # ---- Generate DOE button ----
    if st.button("⚡ Generate 1-Factor DOE", key="lr_generate_doe"):
        if not lr_factor_name:
            st.warning("Please enter the **factor name** first.")
        else:
            # Create evenly spaced factor levels
            x_vals = np.linspace(lr_min, lr_max, int(lr_levels))
            doe_lr = pd.DataFrame({lr_factor_name: x_vals})
            doe_lr.insert(0, "Run", range(1, len(doe_lr) + 1))

            # Add empty response columns
            for r in lr_responses:
                doe_lr[r] = None

            st.subheader("📊 1-Factor DOE (Empty Responses)")
            st.dataframe(doe_lr, hide_index=True)

            # Download CSV
            csv_lr = doe_lr.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download DOE (CSV)",
                csv_lr,
                "linear_regression_doe.csv",
                mime="text/csv",
                key="lr_doe_csv"
            )

            # Download Excel
            out_lr = BytesIO()
            with pd.ExcelWriter(out_lr, engine="openpyxl") as writer:
                doe_lr.to_excel(writer, index=False, sheet_name="DOE")
            st.download_button(
                "📥 Download DOE (Excel)",
                out_lr.getvalue(),
                "linear_regression_doe.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="lr_doe_xlsx"
            )

    # --------------------------------------------------------------
    # 2) UPLOAD COMPLETED DOE & RUN LINEAR REGRESSION + ANOVA
    # --------------------------------------------------------------
    #st.markdown("---")
    st.subheader("📤 Upload Completed DOE for Regression & ANOVA")

    uploaded_file_lr = st.file_uploader(
        "Upload your completed DOE (.csv or .xlsx) with responses filled in",
        type=["csv", "xlsx"],
        key="lr_upload_file"
    )

    data_lr = None
    if uploaded_file_lr is not None:
        try:
            if uploaded_file_lr.name.endswith(".csv"):
                data_lr = pd.read_csv(uploaded_file_lr)
            else:
                data_lr = pd.read_excel(uploaded_file_lr)
            st.success("File loaded successfully!")
            st.dataframe(data_lr.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            data_lr = None

    if data_lr is not None:
        # Numeric columns only
        numeric_cols = data_lr.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("⚠ Need at least **two numeric columns** (one factor, one response).")
        else:
            # Try to use the factor name provided earlier, if present
            if lr_factor_name and lr_factor_name in numeric_cols:
                default_x_idx = numeric_cols.index(lr_factor_name)
            else:
                default_x_idx = 0

            st.subheader("⚙️ Select Factor and Response for Regression")

            x_factor = st.selectbox(
                "Select Factor (X-axis)",
                numeric_cols,
                index=default_x_idx,
                key="lr_x_sel"
            )

            # Candidate responses = other numeric columns
            response_candidates = [c for c in numeric_cols if c != x_factor]

            # Prefer one of the previously defined response names if it exists
            default_y_idx = 0
            for r in lr_responses:
                if r in response_candidates:
                    default_y_idx = response_candidates.index(r)
                    break

            y_response = st.selectbox(
                "Select Response (Y-axis)",
                response_candidates,
                index=default_y_idx if response_candidates else 0,
                key="lr_y_sel"
            )

            if st.button("📊 Run Linear Regression & ANOVA", key="lr_run_reg"):
                # Build model formula
                formula_lr = f"{y_response} ~ {x_factor}"
                st.code(f"Model Formula: {formula_lr}", language="python")

                try:
                    model_lr = ols(formula_lr, data=data_lr).fit()
                    anova_lr = sm.stats.anova_lm(model_lr, typ=2)

                    # ANOVA table
                    st.subheader("📊 ANOVA Table")
                    st.write(anova_lr)

                    # Regression summary
                    st.subheader("📑 Regression Summary")
                    st.text(model_lr.summary())

                    # Residuals / error table
                    residuals_lr = pd.DataFrame({
                        "Fitted": model_lr.fittedvalues,
                        "Residuals": model_lr.resid
                    })
                    st.subheader("📉 Residuals / Error Table")
                    st.dataframe(residuals_lr)

                    

                    # 2) Residuals vs Fitted
                    fig_rvf, ax_rvf = plt.subplots()
                    ax_rvf.scatter(model_lr.fittedvalues, model_lr.resid)
                    ax_rvf.axhline(0, linestyle="--")
                    ax_rvf.set_xlabel("Fitted Values")
                    ax_rvf.set_ylabel("Residuals")
                    ax_rvf.set_title("Residuals vs Fitted")
                    st.pyplot(fig_rvf)

                    # 3) QQ plot
                    fig_qq = sm.qqplot(model_lr.resid, line="45", fit=True)
                    st.pyplot(fig_qq)

                    # ---------- Plots ----------
                    st.subheader("📈 Regression & Diagnostic Plots")
                    fig_scat, ax_scat = plt.subplots()

                    # 1) Scatter + regression line
                    ax_scat.scatter(data_lr[x_factor], data_lr[y_response], label="Data")
                    order = np.argsort(data_lr[x_factor].values)
                    ax_scat.plot(
                        data_lr[x_factor].values[order],
                        model_lr.fittedvalues.values[order],
                        label="Fitted Line"
                    )
                    ax_scat.set_xlabel(x_factor)
                    ax_scat.set_ylabel(y_response)
                    ax_scat.set_title(f"{y_response} vs {x_factor} with Fitted Line")
                    ax_scat.legend()
                    st.pyplot(fig_scat)

                except Exception as e:
                    st.error(f"❌ Linear regression / ANOVA failed: {e}")


