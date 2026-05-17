#!/bin/bash
echo "Looking at the CI failure"
# The failure was: "invalid issue format: "https://github.com/mock/issues/123""
# It happens in .github/workflows/1-create-a-branch.yml because ISSUE_URL is hardcoded to "https://github.com/mock/issues/123" instead of dynamically fetched from the find_exercise job.
# Let's fix this across the workflow.
