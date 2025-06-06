📘 Project Rix: Master Blueprint V55.0 (Hybrid Call Operational, Service Debug Next)

Architecture: Local LangGraph Orchestrator (with MemorySaver Checkpointing for now) successfully making HTTP calls to a deployed Cloud Run Agent Service (rix-classifier-service). Text Embeddings focus.
Date: May 7, 2025 (IST)
Author: Vishal Sharma (Creator, Director)
Compiled By: Gemini (AI Partner, V55.0)
Supersedes: V54.0, and integrates all relevant context.

0. System Instructions for AI Partner (Gemini Role & Directives - V55.0 Focus)

Core Identity: Gemini, AI coding partner for Vishal Sharma.

Primary Role (Dharma): Assist Vishal in debugging the deployed rix-classifier-service on Cloud Run and then systematically implement the remaining hybrid agent calls and service deployments. Ensure the text-based (768d) end-to-end hybrid flow becomes fully operational.

Key Responsibilities:

Cloud Run Debugging: Help diagnose and provide corrected Python code for the rix-classifier-service (main.py) to resolve the AttributeError and any subsequent issues.

Code Generation (Hybrid): Provide complete Python code for:

Modifying other nodes in agents/nodes.py for HTTP calls.

Basic FastAPI/Python wrappers for the remaining Cloud Run agent services (rix-thinker-service, rix-tool-executor-service, etc.).

Cloud Integration: Guide on Cloud Run service updates, log analysis, and ensuring OIDC authentication works correctly once services are stable.

Technical Explanation: Clearly explain Cloud Run logs, FastAPI basics, and the interaction between the local orchestrator and cloud services.

Blueprint Management: Meticulously maintain THIS Master Blueprint V55.0.

Continuity & Context: "Without forget a pin." Retain all history.

Adherence: V52.3 architecture, Text Embeddings (text-embedding-005), V52.0 file structure, config.json.

Constraints:

Immediate Focus: Fixing the rix-classifier-service to correctly process requests from the classifier_node. Then, implementing the next agent service (e.g., rix-thinker-service).

Defer multimodal until text-based hybrid flow is robust.

Communication Style: Clear, step-by-step, complete code, patient debugging.

Mantra: "Debug Cloud Service, Implement Next Hybrid Node, Test End-to-End, Maintain V55.0 Context, Solidify Rix Hybrid V55.0 for Vishal Sharma."

1. Vision Statement (V55.0 - First Hybrid Call Achieved, Solidifying Services)
(Refined from V54.0)

Building upon the successful first hybrid HTTP call from the local Rix LangGraph orchestrator to the deployed rix-classifier-service on Cloud Run. The immediate vision is to debug and stabilize this rix-classifier-service and then systematically implement and deploy the remaining Cloud Run agent services (Thinker, Tool Executor, Memory Writer, Finalizer). The goal is a fully operational, testable, end-to-end hybrid system using Text Embeddings (text-embedding-005, 768d), orchestrated locally via run_rix_cli.py and executing agent logic in the cloud. This will validate the V52.3 architecture and create a robust platform for future multimodal integration (RCD), advanced SCIIC learning, and the broader Proto-AGI objectives.

2. Core Philosophy & Guiding Principles (V55.0)

Local LangGraph Orchestration (with MemorySaver for now): Core brain works, state within a single run is managed. Persistence via Firestore was temporarily paused to unblock, can be revisited.

Cloud Run Agent Services: This is the execution layer. rix-classifier-service is deployed, needs a small fix.

Authenticated HTTP Communication: OIDC token generation is WORKING from classifier_node.

Text Embeddings First (text-embedding-005, 768d): Confirmed and working in initialization.py.

Cloud Services for Data/AI: Cloud SQL/pgvector (vector memory, schema vector(768) CONFIRMED), Vertex AI (for models called by services). Firestore still used by initialization.py for a client instance (V1.9.4 style), but not for LangGraph checkpointing currently.

CLI Testability (run_rix_cli.py): Proven interface, now testing hybrid calls.

Modularity (V52.0 Structure): Stable.

Incremental Hybrid Rollout: One service at a time.

3. Project History Summary (V55.0 - Highlighting Key Milestones & Pivots)

Pre-V52: Exploration, Electron IDE focus, V25.0 defined Async Manager Router.

V52.0-V52.3: Refactoring, shift to V52.3 (Local LangGraph + Cloud Services).

V52.4: Text Embedding unblocking strategy.

V53.0 (The Great Dependency Battle): Extensive environment troubleshooting.

Fresh Python install, clean .venv.

Corrected FirestoreSaver import path identified (from langchain_google_firestore import FirestoreSaver).

V54.0 (Local Graph Operational):

MAJOR BREAKTHROUGH 1: Successfully ran run_rix_cli.py invoking the local LangGraph (rix_graph_app) with Firestore checkpointing working, using corrected import paths and deferred instantiation.

PIVOT (due to persistent ModuleNotFoundError at graph_pipeline.py top-level for Firestore checkpointer when run via script): Switched LangGraph to use MemorySaver for checkpointing (V1.1.5 of graph_pipeline.py, V1.9.7 of initialization.py removing its own checkpointer setup). This resolved the final import/compilation blockers for the graph itself.

MAJOR BREAKTHROUGH 2 (Late May 6th/Early May 7th):

Successfully ran run_rix_cli.py with LangGraph using MemorySaver.

Implemented HTTP call logic (with OIDC token generation) in agents/nodes.py::classifier_node.

Deployed a basic FastAPI rix-classifier-service to Cloud Run.

ACHIEVED FIRST HYBRID CALL: classifier_node successfully called the deployed Cloud Run service. OIDC token generation worked. Service received the call but returned a 503 due to an internal error (NameError: name 'Literal' is not defined in service's main.py).

Cloud Run Service Debug (Ongoing): Identified the NameError in the service. Updated the service's main.py to include from typing import Literal and its requirements.txt to include typing-extensions. Latest Cloud Run logs show a new AttributeError: 'Agent' object has no attribute 'instruction' in the ADK-style code within the service.

4. Current Detailed State (V55.0 - FIRST HYBRID CALL MADE, SERVICE DEBUG)

Python Environment: Stable fresh Python 3.11, clean .venv with minimal requirements.txt (from response #47).

Core Initialization (core/initialization.py V1.9.7 style - no checkpointer init): OPERATIONAL.

LangGraph Orchestration (orchestration/graph_pipeline.py V1.1.5 style - uses MemorySaver): OPERATIONAL. rix_graph_app compiles and runs.

Agent Nodes (agents/nodes.py V1.1.0 style):

classifier_node: Successfully makes an async HTTP POST call with an OIDC token to the deployed rix-classifier-service. Correctly handles the service's error response (e.g., 503 or current AttributeError) and falls back to placeholder logic.

Other nodes (thinker, tool_executor, etc.) are async def placeholders.

CLI (run_rix_cli.py V1.2.3 style - async graph invocation): OPERATIONAL. Successfully initializes Rix, gets the compiled graph, and uses asyncio.run() with rix_graph_app.astream().

Cloud SQL Schema: Confirmed and fixed by Vishal to vector(768). OPERATIONAL.

rix-classifier-service (Cloud Run):

DEPLOYED using FastAPI, from GitHub source, via Cloud Build (Dockerfile method).

Receives HTTP calls from the local classifier_node.

CURRENTLY FAILING INTERNALLY with AttributeError: 'Agent' object has no attribute 'instruction' in its main.py (ADK-style code). The previous NameError: name 'Literal' is not defined issue was resolved.

Utilities (core/utils.py V1.2.0): get_oidc_token() is implemented and WORKING.

Configuration (config.json): Contains the correct URL for the deployed rix-classifier-service.

Key Success Point: The end-to-end path for the first hybrid call (CLI -> Local Graph Node -> OIDC Auth -> HTTP POST -> Deployed Cloud Run Service) is established. The current issue is within the Cloud Run service's application code.

5. What We Want (Goals - V55.0 Perspective)

Immediate Goal (V55.0 Sprint):

Fix rix-classifier-service: Resolve the AttributeError: 'Agent' object has no attribute 'instruction' in its main.py so it can process the request from classifier_node and return a valid classification JSON (even if still based on simple placeholder logic initially).

Achieve Successful End-to-End Hybrid Classification: run_rix_cli.py -> classifier_node -> HTTP Call -> rix-classifier-service processes successfully -> HTTP Response with classification -> classifier_node updates LangGraph State -> Graph continues based on service-provided classification.

Begin implementing the next agent: thinker_node making HTTP calls to a new rix-thinker-service.

Mid-Term & Long-Term Goals: Remain the same as V54.0 (full agent logic in services, RCD, SCIIC, GOD IDE, etc.).

6. Core Logic Flow & classify_input Explanation

Overall Flow: The Mermaid diagram we validated still represents the intended agent interaction sequence.

classify_input (Function in rix-classifier-service/main.py):

Vishal, think of this like a specialized worker at a post office.

Input: It receives a "package" (the HTTP request) from your local classifier_node. This package contains your session_id and the user_input (the message you typed).

Job: Its job is to look at your user_input and decide what kind of task it is. Is it just a simple "CHAT"? Is it an "ASK" for information? Is it a "WORK" request that might need tools? Or is it for the "RCD" cartoon pipeline?

How it does its job (eventually): It will use a powerful AI model (like gemini-1.5-pro-preview-0514 specified as CLASSIFIER_MODEL in your config.json) and a set of instructions (from rix_soul_classifier.json) to make this decision. It's like giving the postal worker a rulebook and a smart helper.

Current (Placeholder) Job: Right now, because we're testing, this worker in rix-classifier-service is just doing a very simple keyword check (e.g., if "file" is in the input, it calls it "WORK"). It's not using the big AI model yet.

Output: After deciding, it prepares a "label" (a JSON response) saying, for example, {"classification": "WORK", "details": "This looks like a work task"} and sends this label back to your local classifier_node.

The Error: The current AttributeError inside this classify_input function means the worker is trying to use a tool or follow an instruction (self.agent.instruction) in a way that's not quite right for the ADK Agent object it has. We need to fix how it uses its "smart helper" (the ADK agent).

7. Component Inventory & File Structure (V55.0 - Key Active Files)

Largely the same as V54.0.

New/Active Service Directory: C:\Rix_Dev\Pro_Rix\cloud_services\rix_classifier_service\ (and its counterpart on GitHub) containing main.py, Dockerfile, requirements.txt.

requirements.txt (for the main project .venv): Successfully installs the minimal set allowing LangGraph (with MemorySaver) and core GCP libs to run.

Rix_Brain/core/initialization.py: V1.9.7 style (no Firestore checkpointer setup, keeps Firestore client init for now).

Rix_Brain/orchestration/graph_pipeline.py: V1.1.5 style (uses MemorySaver, deferred imports working).

Rix_Brain/agents/nodes.py: V1.1.0 style (HTTP call implemented in classifier_node).

run_rix_cli.py: V1.2.3 style (async graph invocation).

8. Tool Calling Mechanism (V55.0 Path)
(No change from V54.0 at this stage, focus is on agent service calls)

9. Memory Management (V55.0 Focus)
(No change from V54.0 at this stage, vector memory still Cloud SQL, will be accessed by services)

10. Checkpointing Mechanism (V55.0)

LangGraph Checkpoints: MemorySaver.

Imported and instantiated directly in orchestration/graph_pipeline.py.

STATUS: CONFIRMED OPERATIONAL. Graph runs without the previous Firestore-related import errors. State is held in RAM for the duration of a run_rix_cli.py execution.

11. Cloud Run Services Plan (V55.0 - rix-classifier-service Debug)
(Focus on fixing the current service)

12. Frontend Architecture Context (V55.0)
(No change from V54.0, CLI is primary)

13. Known Issues & Blockers (V55.0)

PRIMARY BLOCKER: rix-classifier-service on Cloud Run is failing internally with AttributeError: 'Agent' object has no attribute 'instruction' in its main.py when trying to process a request.

Other agent nodes are still placeholders.

Placeholder MemoryWriter/ToolExecutor nodes have temporary local logic.

14. Future Plan & IMMEDIATE PRIORITIES (V55.0)

FIX rix-classifier-service (main.py):

Correct the ADK Agent usage within the /classify endpoint to resolve the AttributeError: 'Agent' object has no attribute 'instruction'. Ensure it uses the agent.run(payload.user_input) method correctly or an equivalent ADK pattern for getting a classification if it's not meant to have an instruction attribute directly.

Ensure the service's requirements.txt (in GitHub and used for Cloud Build) includes typing-extensions if Literal is used (this was likely fixed based on recent logs).

Commit and push changes to GitHub.

Trigger a redeployment of rix-classifier-service on Cloud Run.

Test Successful End-to-End Classification:

Run run_rix_cli.py.

Enter input.

Verify:

classifier_node calls the service.

rix-classifier-service (Cloud Run) logs show it processes the request without internal errors.

Service returns a valid JSON classification (e.g., {"classification": "WORK", ...}).

classifier_node receives this, updates PipelineState.

The rest of the LangGraph (placeholder nodes) proceeds based on this service-provided classification.

Implement rix-thinker-service (Basic):

Create a new directory cloud_services/rix_thinker_service/.

Add main.py (FastAPI, placeholder logic), Dockerfile, requirements.txt.

Push to GitHub.

Deploy to Cloud Run. Get its URL.

Update config.json with RIX_THINKER_SERVICE_URL.

Implement HTTP Call in thinker_node (agents/nodes.py):

Modify thinker_node to call the new rix-thinker-service via HTTP POST with OIDC, similar to classifier_node.

Handle response/fallback.

Test End-to-End (Classifier -> Thinker): Verify the flow.

Repeat for other agent nodes and services.
