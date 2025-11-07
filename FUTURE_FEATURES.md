# Future Features & Improvements

This document tracks potential enhancements for future implementation.

## Session & Chat Improvements

### Timestamp Context for AI
**Priority:** Medium
**Description:** Include timestamps in messages sent to the AI so it can understand temporal context.

**Current behavior:**
- Timestamps stored in session files but not sent to AI
- AI has no awareness of time gaps between messages
- Can be disorienting when resuming sessions after hours/days

**Proposed solution:**
- Add timestamp metadata to message content or system prompt
- Format: "Previous message was from 2 hours ago at 3:15 PM"
- Could be subtle (in system prompt) or explicit (in message)

**Example:**
```
System: You are a helpful assistant. Current time: 5:15 PM Nov 7, 2025
User: [continued from 3:15 PM] What were we discussing?
```

### Show Previous Messages on Session Resume
**Priority:** Medium
**Description:** Display recent conversation history when resuming a session.

**Current behavior:**
- Session loads silently
- User sees blank prompt
- Must use `/history` to see previous context

**Proposed improvement:**
- Show last N messages (e.g., 3-5) when resuming
- Format nicely with timestamps
- Help user remember what they were discussing

**Example:**
```
Resuming session: myproject (24 messages, last active 2 hours ago)

Recent history:
  [3:15 PM] You: Can you help me debug this function?
  [3:16 PM] Assistant: Sure! Let me take a look...
  [3:18 PM] You: It's throwing a KeyError

============================================================
You:
```

## Tool & Harness System

### Coordinator Agent
**Priority:** Low
**Description:** Meta-agent that receives summaries from all active contexts and coordinates between them.

**Use case:**
- IRC bot mentions something relevant to current game
- Coordinator could alert or connect contexts
- Cross-context awareness without breaking isolation

### Queueing & Priority System
**Priority:** Low
**Description:** Handle concurrent requests with priority levels.

**Current:** Sequential processing, first-come-first-served
**Future:** Priority queue (IRC urgent > background game analysis)

## Documentation

### Documentation Consolidation
**Priority:** Medium
**Status:** Noted in SESSION_STORAGE.md

**Issue:** Growing number of specialized docs
**Files:** CLAUDE.md, SESSION_STORAGE.md, DOCKER_SETUP.md, HARNESS_GUIDE.md, etc.

**Proposed:**
- Main README.md (overview, quick start)
- docs/ directory with organized guides
- Index/table of contents
- Cross-references between docs

## Testing

### Pytest Migration
**Priority:** Low
**Current:** Simple shell script runner
**Future:** Proper pytest setup with fixtures, parametrization, coverage

### CI/CD Integration
**Priority:** Low
**Description:** Automated testing on commits/PRs

## Performance

### Session Caching Improvements
**Priority:** Low
**Ideas:**
- Predictive loading (preload likely-next sessions)
- Compression for old sessions
- Archive strategy for sessions > N days old

### vLLM KV Cache Optimization
**Priority:** Low
**Description:** Research if we can manually manage KV cache for better multi-context performance

---

## How to Add Items

When you think of a future feature:
1. Add a section under appropriate category
2. Include: Priority, Description, Current/Proposed behavior
3. Add use cases or examples if helpful
4. Update this file via git (tracks history of ideas)

## How to Implement

When ready to implement:
1. Move item from this file to active TODO list
2. Create detailed plan
3. Remove from this file after completion
