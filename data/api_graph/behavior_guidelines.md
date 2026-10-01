### Rule 1
When a task requires accessing, modifying, or performing actions on user-specific data, private resources, or personalized features within an application, and an authenticated user session is not guaranteed, a login or authentication API is structurally required to establish or verify the user's identity and permissions.

### Rule 2
When a task requires specific, detailed attributes or metrics for individual items within a collection (e.g., play count, duration, specific metadata, artist details), and the API for retrieving the collection itself only provides basic identifiers or summary information, a separate API call to retrieve detailed information for each individual item (e.g., `show_song`, `show_artist`) is required. This also applies when an item is identified by a general description rather than a direct ID, necessitating a search to locate and retrieve its details.

### Rule 3
When a task requires accessing an item within a hierarchical structure (e.g., a file in a directory, a note in a folder), but the exact item name or its precise location is not explicitly provided, or when the user needs to select an item from a set of options, then the capability to list the contents of the relevant container (e.g., `file_system.show_directory`) is needed to discover and locate the target item.

### Rule 4
When a task requires evaluating, selecting, or operating on content that is specifically part of the user's personal library, saved items, or curated collections within an application, rather than newly discovered or public content, an API to access and inspect these personal collections (e.g., `spotify.show_song_library`, `spotify.show_playlist_library`) is required.

### Rule 5
When a task requires operating on the item that is actively being consumed or presented by an application (e.g., the song currently playing), an API to identify and retrieve details of this current item (e.g., `spotify.show_current_song`) is required.

### Rule 6
When a task requires evaluating or modifying existing user-generated content (such as ratings or reviews) for a set of items, an API to retrieve a collection of that content (e.g., `spotify.show_song_reviews`) is required to determine the current state before acting.

### Rule 7
When a task requires understanding or referencing previous communications with a contact, or extracting information from past messages, a capability to search historical messages (e.g., `phone.search_text_messages`) is needed to access that data.
