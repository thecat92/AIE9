<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 15: Build & Serve Agentic Graphs with LangGraph</h1>

| 📰 Session Sheet                                             | ⏺️ Recording                           | 🖼️ Slides                                  | 👨‍💻 Repo    | 📝 Homework                                      | 📁 Feedback                                          |
| ------------------------------------------------------------ | -------------------------------------- | ------------------------------------------- | ------------- | ------------------------------------------------ | ---------------------------------------------------- |
| [Agent Servers](https://github.com/AI-Maker-Space/AIE9/tree/main/00_Docs/Session_Sheets/15_Agent_Servers) |[Recording!](https://us02web.zoom.us/rec/share/lORjByDju6fv4TdE3r93dorY3aNgmSKL_Qk_cX_AMcCQ6cNfSW77unaA1LMVV60.OcI8uEnfVmRAgjSn) <br> passcode: `Dc@&pv1T`| [Session 15 Slides](https://www.canva.com/design/DAG-EJqkRaM/FR3WG_yMA5_BqbWpQlHR9g/edit?utm_content=DAG-EJqkRaM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 15 Assignment: Agent Servers](https://forms.gle/Vb3HNDsyVPQ1jqKX7) | [Feedback 3/3](https://forms.gle/kYmhbVUEMog16mKv8) |

### Prerequisites

Before starting, ensure you have the following:

- **Python 3.11+** installed
- An **OpenAI API Key**
- A **Tavily API Key**
- (Optional) **LangSmith** credentials for tracing

Create a `.env` file in this directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
2. Run `uv sync` to install dependencies.

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API from a different terminal
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangSmith Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

>NOTE: This can be done in place of the Main Assignment

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.

When submitting, provide:
- Your Loom video link demonstrating the MCP server integration
- The GitHub URL to your completed Advanced Build

Have fun!
</details>

### Questions & Activities

#### Question 1:
What is the key architectural difference between the `simple_agent` and `agent_with_helpfulness` graphs? Specifically, explain how the helpfulness evaluation loop works and what mechanisms are in place to prevent it from running indefinitely.

##### Answer:
The main difference between the simple_agent and agent_with_helpfulness graphs is that the second one adds an evaluation step.
The simple_agent has a very direct flow. It receives a user question, may call a tool if needed, generates an answer, and then stops.
The agent_with_helpfulness graph adds an extra step after the answer is generated. A helpfulness node checks if the response actually answers the user’s question. In LangSmith, this appears as a helpfulness result like “HELPFULNESS: Y”.
If the answer is helpful, the workflow ends. If it is not helpful, the system sends the task back to the agent so it can try again and produce a better response. To make sure this loop does not run forever, a maximum retry limit is set. If that limit is reached, the workflow stops even if the answer is still not marked as helpful.
This design shows how an evaluation loop can improve answer quality. The agent generates responses, while a separate step checks their quality and allows a few retries, while still keeping the system controlled and safe.


#### Question 2:
What is the role of `langgraph.json` in the LangGraph Deployments? Describe each of its key fields and how the platform uses this file to discover and serve your graphs.

##### Answer:
The langgraph.json file is the configuration file that tells LangGraph how to find, load, and run agent graphs. It defines where the graphs are located, what dependencies are needed, and how the assistants should be exposed through the LangGraph server.
Important fields control how this works. The version field defines the configuration format. dependencies lists the Python packages or project paths required to run the graphs (for example "." installs the current project). The env field points to the environment file that contains API keys and other settings, and python_version specifies which Python version to use.
The graphs section maps graph IDs to the Python paths where the graphs are defined, while the assistants section defines the assistants that will be available on the server. When the server starts with uv run langgraph dev, LangGraph reads this file and automatically loads the graphs and exposes the assistants in LangSmith Studio.


#### Activity #1:
Create your own agent graph! Build a new graph in `app/graphs/` with a custom evaluation node (e.g., a vibe checker, a fact verifier, a summarizer — get creative!). Register it in `langgraph.json`, serve it with `uv run langgraph dev`

##### Answer:
Custom Agent Graph: Summarization Agent

I built a custom agent graph called summarization_agent that adds a summary evaluation step after the agent generates a response.
The graph includes a summarizer node that checks whether the agent’s output is concise, captures the main points, and is easy to read. If the summary meets the criteria (SUMMARY: Y), the workflow ends. If not, the workflow loops back to the agent so it can produce a better summary.
A loop limit prevents the evaluation from running forever.
The agent was registered in langgraph.json and tested in LangGraph Studio after starting the server with uv run langgraph dev. The summarizer node shows up in the execution trace and evaluates each generated summary.


# Ship 🚢

- The completed notebook.
- 5min. Loom Video

# Share 🚀

- Walk through your notebook and explain what you've completed in the Loom video
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

# Submitting Your Homework

### Main Homework Assignment

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your AIE9 repo:
    - _(You should have completed this process already.)_ For your initial repo setup, see [Initial_Setup](https://github.com/AI-Maker-Space/AIE9/tree/main/00_Docs/Prerequisites/Initial_Setup)
    - To get the latest updates from AI Makerspace into your own AIE9 repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `15_LangGraph_Platform` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Answer Questions 1 - 2 using the `##### Answer:` markdown cell below them in the README
4. Complete Activity #1 in the README
5. Add, commit and push your modified files to your GitHub repository.

When submitting your homework, provide:
- Your Loom video link
- The GitHub URL to the `15_LangGraph_Platform` folder on your assignment branch
