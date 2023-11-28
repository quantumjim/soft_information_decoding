# # Remove all files and directories specified in .gitignore from the remote repository

# # Add all files and directories specified in .gitignore to a list
# files_to_remove=$(git ls-files --others --cached --exclude-standard)

# # Iterate over the list of files to remove
# for file in $files_to_remove; do
#   # Remove the file from the remote repository (origin)
#   git rm --cached "$file"
# done

# # Commit the changes
# git commit -m "Removed files specified in .gitignore from remote"

# # Push the changes to the remote repository
# git push origin main
