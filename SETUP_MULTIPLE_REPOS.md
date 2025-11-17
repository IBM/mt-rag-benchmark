# Setup Multiple Git Remotes

This guide shows how to push to both the original repository and your new repository.

## Current Setup ✅

- **origin**: Points to `Umit-Azirakhmet/mt-rag-benchmark` (original repo)
- **myrepo**: Points to `pratirvce/MTRAG-Eval` (your new repo) ✅ **CONFIGURED**

Both remotes are now set up! You can push to either repository.

## ✅ Already Configured!

Your remotes are already set up:
- **origin**: `Umit-Azirakhmet/mt-rag-benchmark`
- **myrepo**: `pratirvce/MTRAG-Eval` ✅

## Push to Both Repositories

### Push to Your New Repo (Your Repository) ✅
```bash
git push myrepo mtrag_prati_v0
# or push all branches
git push myrepo --all
```

### Push to Original Repo (when you have access)
```bash
git push origin mtrag_prati_v0
```

### Push to Both at Once
```bash
git push origin mtrag_prati_v0 && git push myrepo mtrag_prati_v0
```

### Push Current Branch to Both
```bash
# Get current branch name
BRANCH=$(git branch --show-current)
git push origin $BRANCH && git push myrepo $BRANCH
```

## Step 4: Create New Repository on GitHub (if needed)

If you haven't created the new repository yet:

1. Go to https://github.com/new
2. Repository name: `mt-rag-benchmark` (or any name you prefer)
3. Choose Public or Private
4. **Don't** initialize with README, .gitignore, or license
5. Click "Create repository"
6. Copy the repository URL
7. Use it in Step 1 above

## Quick Reference Commands

```bash
# View all remotes
git remote -v

# Push to your repository (works now!)
git push myrepo mtrag_prati_v0

# Push to original repository (when you have access)
git push origin mtrag_prati_v0

# Push to both repositories
git push origin mtrag_prati_v0 && git push myrepo mtrag_prati_v0

# Push all branches to your repo
git push myrepo --all
```

## Managing Multiple Remotes

### List all remotes
```bash
git remote -v
```

### Remove a remote
```bash
git remote remove myrepo
```

### Change remote URL
```bash
git remote set-url myrepo NEW_URL
```

### Push to specific remote
```bash
git push myrepo branch-name
```

## Recommended Setup

1. **origin**: Original repository (Umit-Azirakhmet)
2. **myrepo** or **backup**: Your personal repository

This way you can:
- Push to your repo anytime: `git push myrepo mtrag_prati_v0`
- Push to original when you have access: `git push origin mtrag_prati_v0`
- Keep both in sync

