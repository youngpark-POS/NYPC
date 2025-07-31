#!/bin/bash
echo "Creating branch practice/jh..."
git checkout -b practice/jh
echo "Branch created. Current branch:"
git branch --show-current
echo "Pushing to remote..."
git push -u origin practice/jh
echo "Branch created and pushed successfully!"