                outputs=task_output
            )

            task_id = gr.Number(
                label="Task ID to complete",
                precision=0
            )

            complete_button = gr.Button("Mark Completed")
            complete_output = gr.Textbox()

            complete_button.click(
                complete_task,
                inputs=task_id,
                outputs=complete_output
            )

        # SEARCH
        with gr.Tab("🔍 Search"):
            keyword = gr.Textbox(label="Search keyword")

            category = gr.Dropdown(
                ["All", "Study", "Hackathon", "Events", "Work", "Home"],
                value="All",
                label="Category"
            )

            priority = gr.Dropdown(
                ["All", "High", "Medium", "Low", "Normal"],
                value="All",
                label="Priority"
            )

            search_button = gr.Button("Search")
            search_output = gr.Dataframe()

            search_button.click(
                advanced_search,
                inputs=[keyword, category, priority],
                outputs=search_output
            )

        # DASHBOARD
        with gr.Tab("📊 Dashboard"):
            dashboard_button = gr.Button("Refresh Dashboard")
            dashboard_output = gr.Markdown()

            dashboard_button.click(
                dashboard,
                outputs=dashboard_output
            )

        # EXPORT
        with gr.Tab("📥 Export"):
            export_button = gr.Button("Export CSV")
            export_output = gr.File()

            export_button.click(
                export_messages,
                outputs=export_output
            )

        # GMAIL
        with gr.Tab("📧 Gmail"):
            gmail_button = gr.Button("View Gmail Setup")
            gmail_output = gr.Markdown()

            gmail_button.click(
                gmail_setup_information,
                outputs=gmail_output
            )

        # TESTING
        with gr.Tab("🧪 Testing"):
            test_button = gr.Button("Run Tests")
            test_output = gr.Textbox(lines=8)

            test_button.click(
                run_tests,
                outputs=test_output
            )

        # DEPLOYMENT
        with gr.Tab("🚀 Deployment"):
            deployment_button = gr.Button(
                "View Deployment Checklist"
            )

            deployment_output = gr.Markdown()

            deployment_button.click(
                deployment_information,
                outputs=deployment_output
            )

    gr.Markdown("""
    ---
    **MindSort AI | Smart Message Management**
    """)


# ============================================================
# 14. START SERVER WITH LOGIN PROTECTION
# ============================================================

if __name__ == "__main__":
    username = os.environ.get("MINDSORT_USERNAME")
    password = os.environ.get("MINDSORT_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "Login credentials are missing. In Render, add the "
            "MINDSORT_USERNAME and MINDSORT_PASSWORD environment "
            "variables, then redeploy."
        )

    if len(password) < 12:
        raise RuntimeError(
            "MINDSORT_PASSWORD must be at least 12 characters long. "
            "Set a longer unique password in Render."
        )

    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        auth=(username, password)
    )
