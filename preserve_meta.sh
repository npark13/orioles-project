#!/bin/bash
# Script used during rebase to ensure commits keep original metadata

orig_author_name=$(git show -s --format='%an' HEAD)
orig_author_email=$(git show -s --format='%ae' HEAD)
orig_author_date=$(git show -s --format='%aD' HEAD)

GIT_AUTHOR_NAME="$orig_author_name" \
GIT_AUTHOR_EMAIL="$orig_author_email" \
GIT_AUTHOR_DATE="$orig_author_date" \
GIT_COMMITTER_NAME="$orig_author_name" \
GIT_COMMITTER_EMAIL="$orig_author_email" \
GIT_COMMITTER_DATE="$orig_author_date" \
git commit --amend --no-edit --allow-empty --allow-empty-message
