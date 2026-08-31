# MCP Server Implementation Guide

## Overview

The PersonalOS project uses the **Model Context Protocol (MCP)** to provide AI agents with external capabilities and tools. MCP servers are modular, reusable components that define what an agent can do.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            MCP Server Manager (Orchestrator)                 │
│  - Routes tool calls to appropriate servers                  │
│  - Manages server lifecycle                                  │
│  - Caching and result aggregation                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌────────┐    ┌────────┐    ┌────────┐
   │ Jobs   │    │ Files  │    │Google  │
   │  MCP   │    │  MCP   │    │  MCP   │
   └────────┘    └────────┘    └────────┘
```

## Core Components

### 1. Base MCP Server (`personalos/mcp/base.py`)

Provides the foundation for all MCP servers.

```python
class MCPServer(ABC):
    """Base class for MCP servers."""
    
    def register_tool(self, schema: ToolSchema, handler: callable)
    async def execute(self, tool_name: str, **kwargs) -> Dict
    def get_tools(self) -> List[Tool]
```

**Key Features:**
- Tool registration with schema validation
- Async-first execution model
- Error handling and logging
- Parameter validation

### 2. Tool Schema (`ToolSchema`)

Defines tool interface using JSON Schema.

```python
schema = ToolSchema(
    name="search_jobs",
    description="Search for jobs",
    parameters={
        "type": "object",
        "properties": {
            "keywords": {"type": "array"},
            "locations": {"type": "array"},
        }
    },
    required=["keywords", "locations"]
)
```

### 3. Cache Layer (`personalos/mcp/cache.py`)

Reduces API calls and improves performance.

**Implementations:**
- `RedisCache` - Distributed cache for production
- `InMemoryCache` - Simple fallback for development

```python
cache = get_cache()
cached_result = await cache.get("mcp:search_jobs:hash123")
await cache.set("mcp:search_jobs:hash123", result, ttl=3600)
```

### 4. MCP Manager (`personalos/mcp/manager.py`)

Orchestrates multiple MCP servers.

```python
manager = get_mcp_manager()

# Execute a tool
result = await manager.execute_tool(
    "search_jobs",
    "jobs",
    keywords=["Python"],
    locations=["Remote"]
)

# List all tools
tools = manager.get_all_tools()
```

## Jobs MCP Server (`mcp_servers/jobs/server.py`)

The primary MCP server for job search operations.

### Available Tools

#### 1. `search_jobs`
Searches job listings across multiple job boards.

**Parameters:**
```json
{
  "keywords": ["Python", "FastAPI"],
  "locations": ["Remote", "NYC"],
  "job_type": "full-time",
  "limit": 50
}
```

**Response:**
```json
{
  "total": 50,
  "jobs": [
    {
      "id": "job_123",
      "title": "Senior Python Developer",
      "company": "TechCorp",
      "location": "Remote",
      "salary": "$120,000 - $160,000",
      "url": "https://example.com/jobs/job_123",
      "relevance_score": 0.95
    }
  ]
}
```

#### 2. `scrape_job_details`
Extracts detailed information from a job posting.

**Parameters:**
```json
{
  "job_id": "job_123",
  "job_url": "https://example.com/jobs/job_123"
}
```

**Response:**
```json
{
  "id": "job_123",
  "title": "Senior Python Developer",
  "description": "Full job description...",
  "requirements": ["5+ years Python", "FastAPI", "Docker"],
  "benefits": ["Health insurance", "Remote", "401k"],
  "skills": ["Python", "FastAPI", "Docker"],
  "experience_level": "senior",
  "difficulty_match": 0.92
}
```

#### 3. `filter_jobs`
Filters and ranks job listings.

**Parameters:**
```json
{
  "jobs": [...],
  "salary_min": 100000,
  "salary_max": 150000,
  "experience_level": "senior",
  "remote_only": true
}
```

**Response:**
```json
{
  "total_filtered": 15,
  "total_input": 50,
  "jobs": [...],
  "filters_applied": {...}
}
```

#### 4. `save_favorite_job`
Saves a job to favorites.

**Parameters:**
```json
{
  "job_id": "job_123",
  "notes": "Interesting opportunity with good benefits"
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "job_123",
  "saved_at": "2026-08-30T12:00:00Z"
}
```

## Integration with Executor

The `JobSearchExecutor` uses MCP servers to perform job searches.

```python
executor = JobSearchExecutor(repo)

# The executor internally:
# 1. Calls search_jobs via MCP
# 2. Scrapes top 20 results in parallel
# 3. Filters and ranks via MCP
# 4. Returns top 10 matches
result_job = await executor.run_job_search(job)
```

### Execution Flow

```python
async def run_job_search(self, job: Job):
    # Step 1: Prepare
    # Step 2: Search via MCP
    search_result = await self.mcp_manager.execute_tool(
        "search_jobs", "jobs",
        keywords=job.keywords,
        locations=job.locations
    )
    
    # Step 3: Scrape details via MCP
    for job in search_results[:20]:
        detail = await self.mcp_manager.execute_tool(
            "scrape_job_details", "jobs",
            job_id=job.id,
            job_url=job.url
        )
    
    # Step 4: Filter via MCP
    filtered = await self.mcp_manager.execute_tool(
        "filter_jobs", "jobs",
        jobs=detailed_results,
        salary_min=job.salary_min,
        salary_max=job.salary_max
    )
    
    return filtered[:10]
```

## Creating a Custom MCP Server

### Template

```python
from personalos.mcp.base import MCPServer, ToolSchema

class CustomMCPServer(MCPServer):
    def __init__(self):
        super().__init__("custom", "Description")
        self.initialize()

    def initialize(self):
        # Define tools
        tool_schema = ToolSchema(
            name="my_tool",
            description="What it does",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                },
            },
            required=["param1"]
        )
        self.register_tool(tool_schema, self._my_tool)

    async def _my_tool(self, param1: str):
        # Implement tool logic
        return {"result": "value"}
```

### Register in Manager

```python
from personalos.mcp.manager import get_mcp_manager

manager = get_mcp_manager()
custom_server = CustomMCPServer()
manager.register_server(custom_server)
```

## Testing MCP Servers

### Unit Tests

```python
@pytest.mark.asyncio
async def test_search_jobs():
    server = JobsMCPServer()
    result = await server.execute(
        "search_jobs",
        keywords=["Python"],
        locations=["Remote"]
    )
    assert result["success"] is True
    assert len(result["result"]["jobs"]) > 0
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_executor_with_mcp():
    executor = JobSearchExecutor(repo)
    job = Job(
        title="Search",
        keywords=["Python"],
        locations=["Remote"]
    )
    result = await executor.run_job_search(job)
    assert result.status == JobStatus.COMPLETED
```

## Best Practices

### 1. Caching
Cache expensive API calls to reduce costs and latency.

```python
cache = get_cache()
result = await cache.get(cache_key)
if result is None:
    result = await api_call()
    await cache.set(cache_key, result, ttl=3600)
```

### 2. Error Handling
Always handle errors gracefully with informative messages.

```python
try:
    result = await external_api.call()
except Exception as e:
    logger.error(f"API call failed: {e}")
    return {"success": False, "error": str(e)}
```

### 3. Rate Limiting
Implement rate limiting to respect API quotas.

```python
rate_limiter = RateLimiter(calls_per_minute=60)
await rate_limiter.acquire()
result = await api_call()
```

### 4. Logging
Log all tool executions for debugging and monitoring.

```python
logger.info(f"Executing tool '{tool_name}' with params {kwargs}")
logger.error(f"Tool execution failed: {error}")
```

## Future Enhancements

1. **Parallel Execution** - Run multiple tool calls concurrently
2. **Circuit Breaker** - Handle API failures gracefully
3. **Streaming** - Support streaming responses for large result sets
4. **Authentication** - Secure tool access with API keys/OAuth
5. **Metrics** - Track tool usage and performance
6. **Versioning** - Support multiple versions of tools

## Configuration

Set via `.env`:
```
# Redis for caching
REDIS_URL=redis://localhost:6379/0

# MCP Servers
MCP_JOBS_ENABLED=true
MCP_GOOGLE_ENABLED=false
MCP_FILES_ENABLED=true
```

## See Also

- [Jobs MCP Server Implementation](../mcp_servers/jobs/server.py)
- [MCP Manager](../personalos/mcp/manager.py)
- [Job Search Executor](../personalos/executor/job_search.py)
- [MCP Tests](../tests/unit/test_mcp_server.py)
