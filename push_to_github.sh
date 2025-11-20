#!/bin/bash
# Helper script to push repository to GitHub
# Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME [REPO_NAME]

set -e

if [ -z "$1" ]; then
    echo "Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME [REPO_NAME]"
    echo "Example: ./push_to_github.sh johndoe"
    echo "Example: ./push_to_github.sh johndoe my-repo-name"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME=${2:-nlp-domain-alignment}

echo "Setting up remote for GitHub repository..."
echo "Repository: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""
echo "⚠️  IMPORTANT: Make sure you've created the repository on GitHub first!"
echo "   Go to: https://github.com/new"
echo "   Repository name: ${REPO_NAME}"
echo "   DO NOT initialize with README, .gitignore, or license"
echo ""
read -p "Press Enter once you've created the repository on GitHub..."

# Check if remote already exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "Remote 'origin' already exists. Updating..."
    git remote set-url origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
else
    echo "Adding remote 'origin'..."
    git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
fi

echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo "   View your repository at: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"

