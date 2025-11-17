# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:12:46 2024

@author: Hemant
"""

import json
from github import Github

def read_config():
    try:
        with open('./config/gitConfig.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Configuration file 'gitConfig.json' not found. Let's create one.")
        config = create_config()
    return config

def create_config():
    github_username = input("Enter your GitHub username: ")
    github_repository = input("Enter your GitHub repository name: ")
    github_token = input("Enter your GitHub personal access token: ")
    excel_file_path = input("Enter path to the Excel file in the repository: ")
    branch_name = input("Enter the branch name where the Excel file is located (e.g., 'main'): ")
    config = {
        "username": github_username,
        "repository": github_repository,
        "token": github_token,
        "git_file_path": excel_file_path,
        "branch_name": branch_name
        }
    with open('./config/gitConfig.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("Configuration file 'gitConfig.json' created successfully.")
    return config

def upload_to_github(file_content, commit_message):
    config = read_config()
    username = config.get('username')
    repository_name = config.get('repository')
    github_token = config.get('token')
    branch_name = config.get('branch_name')
    git_file_path = config.get('git_file_path')
    try:
        g = Github(github_token)
        user = g.get_user(username)
        repo = user.get_repo(repository_name)
    except Exception as e:
        print(f"Error accessing GitHub repository: {str(e)}")
    try:
        contents = repo.get_contents(git_file_path, ref=branch_name)
        repo.update_file(contents.path, commit_message, file_content, contents.sha, branch=branch_name)
        print(f"File '{git_file_path}' updated successfully on GitHub.")
    except Exception as e:
        print(f"Failed to update file '{git_file_path}' on GitHub: {str(e)}")

def Uplode_Latest_Insights(filename):
    commit_message = "Uplode Latest Insights"
    with open(filename, 'rb') as f:
        file_content = f.read()
    res = upload_to_github(file_content, commit_message)
    return res
