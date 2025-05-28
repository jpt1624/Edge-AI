#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building University of South Alabama Edge AI Marp presentations...${NC}"

# Make sure directories exist
THEME_DIR="./theme"
OUTPUT_DIR="./output"

if [ ! -d "$THEME_DIR" ]; then
    echo -e "${YELLOW}Creating theme directory...${NC}"
    mkdir -p "$THEME_DIR"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Creating output directory...${NC}"
    mkdir -p "$OUTPUT_DIR"
fi

# Check for the logo file and handle it properly
LOGO_FILE="$THEME_DIR/usa_logo.png"
if [ ! -f "$LOGO_FILE" ]; then
    echo -e "${YELLOW}Logo not found in theme directory, looking elsewhere...${NC}"
    
    # Try to find logo in current directory
    if [ -f "./usa_logo.png" ]; then
        echo -e "${YELLOW}Found logo in current directory, copying to theme folder...${NC}"
        cp "./usa_logo.png" "$LOGO_FILE"
        echo -e "${GREEN}Logo copied successfully!${NC}"
    else
        echo -e "${RED}Logo not found. Creating a placeholder logo...${NC}"
        # Create a simple placeholder using convert (ImageMagick)
        if command -v convert &> /dev/null; then
            echo -e "${YELLOW}Creating placeholder logo with ImageMagick...${NC}"
            convert -size 200x200 xc:white -fill "#00264C" -gravity center \
                -font Arial -pointsize 40 -annotate 0 "USA" "$LOGO_FILE"
            echo -e "${GREEN}Placeholder logo created!${NC}"
        else
            echo -e "${RED}ImageMagick not found. Please add a logo file manually.${NC}"
            echo -e "${RED}Place usa_logo.png in the $THEME_DIR directory.${NC}"
        fi
    fi
fi

# Copy the theme CSS file
THEME_FILE="$THEME_DIR/usa-theme.css"

# Process all markdown files in the current directory
echo -e "${GREEN}Building presentations with University of South Alabama branding...${NC}"
for file in *.md; do
    if [ -f "$file" ]; then
        # Get the filename without extension
        filename=$(basename -- "$file")
        filename="${filename%.*}"
        
        echo -e "${GREEN}Converting $file to $filename.pptx with USA branding...${NC}"
        
        # Run Marp to convert to PowerPoint with absolute resource path
        marp --theme "$THEME_FILE" --allow-local-files \
            --html --output "$OUTPUT_DIR/$filename.pptx" "$file"
        
        # Check if the conversion was successful
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully converted $file to $OUTPUT_DIR/$filename.pptx${NC}"
        else
            echo -e "${RED}Error converting $file - please check the error messages above${NC}"
            echo -e "${YELLOW}Trying alternative method...${NC}"
            
            # Try alternative with different options
            marp --theme-set "$THEME_DIR" --allow-local-files \
                --html --output "$OUTPUT_DIR/$filename.pptx" "$file"
                
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Alternative method successful!${NC}"
            else
                echo -e "${RED}Both conversion methods failed.${NC}"
            fi
        fi
    fi
done

echo -e "${GREEN}All University of South Alabama Edge AI presentations built!${NC}"
echo -e "${YELLOW}Presentations are available in the $OUTPUT_DIR directory${NC}"