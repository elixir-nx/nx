# Command to generate the compilation database file.
bazel build exla:exla_compdb

# Location of the compilation database file.
outfile="$(bazel info bazel-bin)/exla/compile_commands.json"

# Command to replace the marker for exec_root in the file.
execroot=$(bazel info execution_root)
sed -i.bak "s@__EXEC_ROOT__@${execroot}@" "${outfile}"

# The compilation database is now ready to use at this location.
echo "Compilation Database: ${outfile}"