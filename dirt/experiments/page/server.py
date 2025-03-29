from livereload import Server, shell

PORT = 4000

# Create a server instance
server = Server()

# Watch all files in the current directory for changes
server.watch(".", delay=0.5)  # Adjust delay as needed

# Serve the directory
server.serve(port=PORT, host="localhost")

print(f"Serving with live reload at http://localhost:{PORT}")
