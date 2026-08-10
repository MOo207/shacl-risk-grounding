#!/bin/bash
#
# Script to load GRC Ontology and SHACL shapes into Neo4j
# Uses neosemantics plugin for RDF/SHACL support
#
# Prerequisites:
# 1. Neo4j 4.x or 5.x installed and running
# 2. neosemantics plugin installed: https://github.com/jbarrasa/neosemantics
# 3. Neo4j credentials configured below
#

# Configuration
NEO4J_BOLT_URI="${NEO4J_BOLT_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONTOLOGY_DIR="$SCRIPT_DIR/ttl"
SHAPES_DIR="$SCRIPT_DIR/shapes"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "GRC Ontology and SHACL Loader"
echo "=========================================="
echo ""
echo "Neo4j BOLT URI: $NEO4J_BOLT_URI"
echo "Neo4j User: $NEO4J_USER"
echo ""

# Function to execute Cypher query
execute_cypher() {
    local query="$1"
    cypher-shell -a "$NEO4J_BOLT_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "$query" 2>/dev/null
}

# Check if Neo4j is accessible
echo -n "Checking Neo4j connection... "
if execute_cypher "RETURN 1 AS test" | grep -q "test"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Error: Cannot connect to Neo4j at $NEO4J_BOLT_URI"
    echo "Please check:"
    echo "  1. Neo4j is running"
    echo "  2. neosemantics plugin is installed"
    echo "  3. Credentials are correct"
    exit 1
fi

echo ""
echo "Step 1: Loading Ontology..."
echo "---------------------------"

# Load main ontology
if [ -f "$ONTOLOGY_DIR/grc-ontology.ttl" ]; then
    echo -n "Loading grc-ontology.ttl... "
    execute_cypher "CALL n10s.rdf.import.fetch('file://$ONTOLOGY_DIR/grc-ontology.ttl','Turtle')" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}WARNING${NC}: grc-ontology.ttl not found"
fi

echo ""
echo "Step 2: Loading SHACL Shapes..."
echo "--------------------------------"

# Load technical entities SHACL
if [ -f "$SHAPES_DIR/technical-entities.shacl.ttl" ]; then
    echo -n "Loading technical-entities.shacl.ttl... "
    execute_cypher "CALL n10s.rdf.import.fetch('file://$SHAPES_DIR/technical-entities.shacl.ttl','Turtle')" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}WARNING${NC}: technical-entities.shacl.ttl not found"
fi

# Load GRC entities SHACL
if [ -f "$SHAPES_DIR/grc-entities.shacl.ttl" ]; then
    echo -n "Loading grc-entities.shacl.ttl... "
    execute_cypher "CALL n10s.rdf.import.fetch('file://$SHAPES_DIR/grc-entities.shacl.ttl','Turtle')" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}WARNING${NC}: grc-entities.shacl.ttl not found"
fi

echo ""
echo "Step 3: Configuring neosemantics..."
echo "------------------------------------"

# Set neosemantics configuration
echo -n "Setting graph prefix to grc... "
execute_cypher "CALL n10s.graphconfig.init({handleVocabUris: 'MAP', handleMultival: 'ARRAY', handleRDFTypes: 'NODES', handleLang: 'KEEP'})" >/dev/null 2>&1
echo -e "${GREEN}OK${NC}"

echo ""
echo "Step 4: Applying SHACL Constraints..."
echo "-----------------------------------"

# Apply SHACL shapes as validation rules
echo -n "Enabling SHACL validation... "
execute_cypher "CALL n10s.validation.shacl.enable()" >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC}: SHACL may already be enabled or not available"
fi

echo ""
echo "Step 5: Verification..."
echo "------------------------"

# Count loaded nodes
echo "Counting loaded nodes..."
node_count=$(execute_cypher "MATCH (n) RETURN count(n) AS count" | grep -oP '\d+(?=\n)' | tail -1)
echo "Total nodes: $node_count"

# Count loaded relationships
echo "Counting loaded relationships..."
rel_count=$(execute_cypher "MATCH ()-[r]->() RETURN count(r) AS count" | grep -oP '\d+(?=\n)' | tail -1)
echo "Total relationships: $rel_count"

# Show sample nodes
echo ""
echo "Sample nodes:"
execute_cypher "MATCH (n) RETURN labels(n) AS labels, properties(n) AS props LIMIT 5" | head -20

echo ""
echo "=========================================="
echo -e "${GREEN}Loading Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Load your RDF data using: CALL n10s.rdf.import.fetch('file://<path>', 'Turtle')"
echo "2. Validate data with: CALL n10s.validation.shacl.validate()"
echo "3. Query the graph with Cypher"
echo ""
echo "For more information, see ontology/README.md"
