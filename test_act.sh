#!/bin/bash
cat << 'JSON' > event.json
{
  "action": "opened",
  "issue": {
    "number": 1,
    "title": "Introduction to GitHub",
    "user": { "login": "Night-Hawkeye" },
    "state": "open"
  },
  "repository": {
    "name": "skills-introduction-to-github",
    "full_name": "Night-Hawkeye/skills-introduction-to-github",
    "owner": {
      "login": "Night-Hawkeye"
    }
  },
  "sender": {
    "login": "Night-Hawkeye"
  }
}
JSON
./bin/act -j check_step_work -e event.json -W .github/workflows/1-create-a-branch.yml --secret GITHUB_TOKEN=${GITHUB_TOKEN:-""} -P ubuntu-latest=catthehacker/ubuntu:full-latest
