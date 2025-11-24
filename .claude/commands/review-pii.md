# Review for Personal Information and Sensitive Data

Conduct a comprehensive security audit of the codebase to identify any personal information, credentials, or sensitive data.

## Scope

Check all files, with special focus on:
- Recently modified or created files
- Configuration files (.env, .toml, .json, .yaml)
- Source code files (.py, .js, .ts)
- Documentation files (.md)
- Git commit messages and history

## Items to Check

### Critical (Must Fix)
- **API keys or tokens**: Hardcoded credentials
- **Passwords or secrets**: Any authentication credentials
- **Database connection strings**: With embedded credentials
- **Private keys**: SSH keys, SSL certificates, etc.

### High Priority
- **Email addresses**: Personal or organizational emails
- **Project IDs**: GCP project IDs, AWS account IDs, etc.
- **Internal hostnames**: Company-specific URLs or servers
- **User/organization names**: In file paths, comments, or code

### Medium Priority
- **Personal names**: Author names in non-standard locations
- **Phone numbers**: Any contact information
- **IP addresses**: Internal network IPs
- **Custom domain names**: Organization-specific domains

### Low Priority (Context-Dependent)
- **Generic examples**: "example.com", "user@example.com"
- **Public documentation references**: Public API endpoints
- **Standard placeholders**: "YOUR_API_KEY", "TODO: add your key"

## Output Format

For each finding, provide:

1. **Severity**: Critical | High | Medium | Low
2. **File**: Full path and line number
3. **Content**: Quote of the sensitive data (masked if necessary)
4. **Type**: Category of sensitive data
5. **Recommendation**:
   - Remove and add to .gitignore
   - Move to environment variables
   - Replace with placeholder
   - Acceptable (with justification)

## Example Output

```
### Critical Findings: 0

### High Priority Findings: 1

**[HIGH] API Key in Configuration**
- File: `src/config/settings.py:15`
- Content: `API_KEY = "sk-1234567890abcdef"`
- Type: Hardcoded API key
- Recommendation: Remove immediately. Use environment variable `GOOGLE_API_KEY` instead.

### Medium Priority Findings: 0

### Low Priority Findings: 1

**[LOW] Example Email**
- File: `README.md:45`
- Content: `Contact: support@example.com`
- Type: Email address (generic example)
- Recommendation: Acceptable - this is a documentation placeholder.
```

## Summary

Provide a final summary with:
- Total findings by severity
- Overall risk assessment
- Immediate actions required
- Long-term recommendations
