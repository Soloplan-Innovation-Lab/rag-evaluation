import pandas as pd
import plotly.express as px
import streamlit as st

def render_basic_chat_charts():
    st.subheader("Response Details")
    tabs = st.tabs(["Response", "Documents", "Metadata", "Charts"])
    if "response_data" in st.session_state:
        with tabs[0]:
            st.json(st.session_state.response_data)
        with tabs[1]:
            st.json(st.session_state.response_data.get("documents", {}))
        with tabs[2]:
            st.json(st.session_state.response_data.get("steps", {}))
        with tabs[3]:
            rd = st.session_state.response_data
            # Create two columns
            col1, col2 = st.columns(2)

            # Token Usage Charts
            if rd.get("token_usage"):
                token_usage = rd["token_usage"]
                token_usage_df = pd.DataFrame(
                    {
                        "Type": ["Prompt Tokens", "Completion Tokens", "Total Tokens"],
                        "Tokens": [
                            token_usage["prompt_tokens"],
                            token_usage["completion_tokens"],
                            token_usage["total_tokens"],
                        ],
                    }
                )
                with col1:
                    st.subheader("Token Usage")
                    st.bar_chart(token_usage_df.set_index("Type"))

            # Retrieval Steps Duration Charts
            if rd.get("steps"):
                steps = rd["steps"]
                steps_data = []
                for i, step in enumerate(steps):
                    steps_data.append(
                        {
                            "Step": f"Step {i+1} - Pre-Retrieval",
                            "Duration": step["pre_retrieval_duration"],
                        }
                    )
                    steps_data.append(
                        {
                            "Step": f"Step {i+1} - Retrieval",
                            "Duration": step["retrieval_duration"],
                        }
                    )
                    steps_data.append(
                        {
                            "Step": f"Step {i+1} - Post-Retrieval",
                            "Duration": step["post_retrieval_duration"],
                        }
                    )
                # Add the response duration
                steps_data.append(
                    {"Step": "Response Creation", "Duration": rd["response_duration"]}
                )
                steps_df = pd.DataFrame(steps_data)
                with col2:
                    st.subheader("Retrieval Steps Duration")
                    pie_chart = px.pie(
                        steps_df,
                        values="Duration",
                        names="Step",
                    )
                    st.plotly_chart(pie_chart)

            evaluation_data = st.session_state.get("evaluation_data", {})
            if evaluation_data:
                rows = []

                # Assuming evaluation_data itself is a single dictionary with necessary keys
                # Check and process deepeval metrics
                deepeval_metrics = evaluation_data.get("deepeval", {})
                for metric_name, metric in deepeval_metrics.items():
                    rows.append(
                        {
                            "Metric": metric_name,
                            "Reason": metric.get("reason", ""),
                            "Score": metric.get("score", 0),
                            "Threshold": metric.get("threshold", None),
                            "Success": metric.get("success", None),
                        }
                    )

                # Check and process ragas metrics
                ragas_metrics = evaluation_data.get("ragas", {})
                for metric_name, score in ragas_metrics.items():
                    rows.append(
                        {
                            "Metric": metric_name,
                            "Reason": "RAGAS",
                            "Score": score,
                            "Threshold": None,
                            "Success": None,
                        }
                    )

                # Create a DataFrame from the list of rows
                ev_df = pd.DataFrame(rows)

                st.subheader("Evaluation Metrics")
                # Display the DataFrame in Streamlit
                st.dataframe(ev_df, use_container_width=True)

            # Line Graphs for Performance and Token Usage Over Time
            historical_responses = st.session_state.historical_responses

            if historical_responses:
                performance_data = []
                token_usage_data = []

                for i, response in enumerate(historical_responses):
                    if response.get("response_duration"):
                        performance_data.append(
                            {
                                "Request": i + 1,
                                "Metric": "Response creation",
                                "Duration": response["response_duration"],
                            }
                        )

                    if response.get("steps"):
                        steps = response["steps"]
                        for j, step in enumerate(steps):
                            if step.get("pre_retrieval_duration"):
                                performance_data.append(
                                    {
                                        "Request": i + 1,
                                        "Metric": f"Step {j + 1} - Pre-Retrieval",
                                        "Duration": step["pre_retrieval_duration"],
                                    }
                                )
                            if step.get("retrieval_duration"):
                                performance_data.append(
                                    {
                                        "Request": i + 1,
                                        "Metric": f"Step {j + 1} - Retrieval",
                                        "Duration": step["retrieval_duration"],
                                    }
                                )
                            if step.get("post_retrieval_duration"):
                                performance_data.append(
                                    {
                                        "Request": i + 1,
                                        "Metric": f"Step {j + 1} - Post-Retrieval",
                                        "Duration": step["post_retrieval_duration"],
                                    }
                                )

                    if response.get("token_usage"):
                        token_usage = response["token_usage"]
                        token_usage_data.append(
                            {
                                "Request": i + 1,
                                "Type": "Prompt Tokens",
                                "Tokens": token_usage["prompt_tokens"],
                            }
                        )
                        token_usage_data.append(
                            {
                                "Request": i + 1,
                                "Type": "Completion Tokens",
                                "Tokens": token_usage["completion_tokens"],
                            }
                        )
                        token_usage_data.append(
                            {
                                "Request": i + 1,
                                "Type": "Total Tokens",
                                "Tokens": token_usage["total_tokens"],
                            }
                        )

                if performance_data:
                    performance_df = pd.DataFrame(performance_data)
                    performance_df = performance_df.sort_values(by="Duration")
                    st.subheader("Performance Over Time")
                    line_chart = px.area(
                        performance_df,
                        x="Request",
                        y="Duration",
                        color="Metric",
                        title="Response Duration Over Time",
                    )
                    line_chart.update_traces(mode="lines+markers")  # Add data points
                    st.plotly_chart(line_chart)

                if token_usage_data:
                    token_usage_df = pd.DataFrame(token_usage_data)
                    token_usage_df = token_usage_df.sort_values(by="Tokens")
                    st.subheader("Token Usage Over Time")
                    line_chart = px.area(
                        token_usage_df,
                        x="Request",
                        y="Tokens",
                        color="Type",
                        title="Token Usage Over Time",
                    )
                    line_chart.update_traces(mode="lines+markers")  # Add data points
                    st.plotly_chart(line_chart)

    else:
        with tabs[0]:
            st.write("Response will be displayed here...")
        with tabs[1]:
            st.write("Documents will be displayed here...")
        with tabs[2]:
            st.write("Metadata will be displayed here...")
        with tabs[3]:
            st.write("Charts will be displayed here...")
