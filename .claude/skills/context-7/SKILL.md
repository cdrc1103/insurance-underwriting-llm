---
name: context7
description: Retrieves up-to-date documentation and code examples for any programming library. Use when implementing features with external libraries, troubleshooting library errors, or when unsure about correct API usage.
---

When using Context7 for documentation lookup:

1. **Resolve the library ID first**: Always call `resolve-library-id` before `query-docs` to get the correct Context7-compatible library ID
2. **Be specific with queries**: Use detailed questions like "How to set up authentication with JWT in Express.js" rather than vague terms like "auth"
3. **Verify API usage**: When code fails or behaves unexpectedly, query Context7 to check for breaking changes or deprecated methods
4. **Limit calls**: Do not call either tool more than 3 times per question - use the best result available

Common use cases:
- Implementing new features with unfamiliar libraries
- Debugging errors related to library APIs
- Checking for current best practices and patterns
- Verifying method signatures and parameters
