# University of South Alabama - Edge AI Course Presentation

This repository contains Marp presentation files for the "Deploying Edge AI" course at the University of South Alabama. The presentation is built using Marp (Markdown Presentation Ecosystem) with a custom theme matching the University of South Alabama brand colors.

## Repository Contents

- `edge-ai-presentation.md` - The main Markdown file containing the presentation
- `usa-theme.css` - Custom CSS theme file for University of South Alabama styling
- `usa_logo.png` - University of South Alabama logo for the presentation

## Prerequisites

Before building the presentation, make sure you have:

- **Node.js** (version 12 or higher)
- **npm** (Node Package Manager, comes with Node.js)

You can verify your installation by running:
```bash
node --version
npm --version
```

## Installing Marp CLI

Marp CLI is required to convert the Markdown presentation into PowerPoint, PDF, or HTML formats.

### Global Installation (Recommended)

```bash
npm install -g @marp-team/marp-cli
```

Verify the installation:
```bash
marp --version
```

### Project-Specific Installation (Alternative)

If you prefer a local installation for this project only:

```bash
# Initialize a new npm project if not done already
npm init -y

# Install Marp CLI as a development dependency
npm install --save-dev @marp-team/marp-cli
```

With this approach, you'll use `npx` to run Marp:
```bash
npx marp your-presentation.md
```

## Building the Presentation

### Building PowerPoint (PPTX)

```bash
marp --theme-set ./usa-theme.css --allow-local-files --output usa-edge-ai.pptx edge-ai-presentation.md
```

### Building PDF

```bash
marp --theme-set ./usa-theme.css --allow-local-files --output usa-edge-ai.pdf edge-ai-presentation.md
```

### Building HTML

```bash
marp --theme-set ./usa-theme.css --allow-local-files --output usa-edge-ai.html edge-ai-presentation.md
```

The `--allow-local-files` flag is essential as it permits Marp to access local files like the University of South Alabama logo.

## Using with VS Code

If you prefer using Visual Studio Code:

1. Install the "Marp for VS Code" extension
2. Open your `edge-ai-presentation.md` file
3. Add the VS Code Marp theme path in settings:
   ```json
   "markdown.marp.themes": [
     "./usa-theme.css"
   ]
   ```
4. Click the "Open Preview to the Side" button to see a live preview
5. Export using the VS Code commands (Ctrl+Shift+P → "Marp: Export")

## Customizing the Presentation

### Slide Types

The presentation supports different slide types using the `<!-- _class: type -->` directive:

- `<!-- _class: title -->` - For the title slide
- `<!-- _class: section -->` - For section divider slides
- Regular slides (no class needed)

Example:
```markdown
---
marp: true
theme: usa-theme
paginate: true
---

<!-- _class: title -->

# Main Title
## Subtitle

---

<!-- _class: section -->

# Section Title

---

# Regular Slide
- Bullet point 1
- Bullet point 2
```

### Special Elements

The theme includes special styled elements:

```markdown
<div class="highlight-box">
This content will appear in a blue-themed highlight box
</div>

<div class="alert-box">
This content will appear in a red-themed alert box
</div>

<div class="columns">
<div>

## Left Column
Content for the left column

</div>
<div>

## Right Column
Content for the right column

</div>
</div>
```

## Additional Resources

- [Official Marp Documentation](https://marpit.marp.app/)
- [Marp CLI GitHub Repository](https://github.com/marp-team/marp-cli)
- [Markdown Guide](https://www.markdownguide.org/)
