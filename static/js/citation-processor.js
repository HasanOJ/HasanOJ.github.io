// Citation processor for Hugo posts
// This script converts [@citekey] format to inline citations

class CitationProcessor {
    constructor(bibData) {
        this.bibliography = this.parseBibTeX(bibData);
        this.citedKeys = new Set();
    }

    parseBibTeX(bibData) {
        const entries = {};
        const entryRegex = /@\w+\{([^,]+),[\s\S]*?\n\}/g;
        
        let match;
        while ((match = entryRegex.exec(bibData)) !== null) {
            const key = match[1];
            const entry = match[0];
            
            entries[key] = {
                key: key,
                raw: entry,
                author: this.extractField(entry, 'author'),
                title: this.extractField(entry, 'title'),
                year: this.extractField(entry, 'year'),
                journal: this.extractField(entry, 'journal'),
                publisher: this.extractField(entry, 'publisher'),
                doi: this.extractField(entry, 'doi'),
                url: this.extractField(entry, 'url')
            };
        }
        
        return entries;
    }

    extractField(entry, fieldName) {
        const regex = new RegExp(`${fieldName}\\s*=\\s*[{"]([^}"]*)[}"]`, 'i');
        const match = entry.match(regex);
        return match ? match[1].replace(/[{}]/g, '') : '';
    }

    formatInlineCitation(key) {
        const entry = this.bibliography[key];
        if (!entry) {
            return `[${key}]`;
        }

        this.citedKeys.add(key);
        
        let author = entry.author;
        if (author.includes(' and ')) {
            const authors = author.split(' and ');
            if (authors.length > 2) {
                author = `${authors[0].split(',')[0]} et al.`;
            } else {
                author = `${authors[0].split(',')[0]} & ${authors[1].split(',')[0]}`;
            }
        } else {
            author = author.split(',')[0];
        }

        return `(<a href="#ref-${key}">${author}, ${entry.year}</a>)`;
    }

    processCitations(content) {
        const citationRegex = /\[@([^\]]+)\]/g;
        
        return content.replace(citationRegex, (match, key) => {
            return this.formatInlineCitation(key);
        });
    }

    generateBibliography() {
        if (this.citedKeys.size === 0) {
            return '';
        }

        let bibliography = '\n\n## References\n\n';
        
        for (const key of Array.from(this.citedKeys).sort()) {
            const entry = this.bibliography[key];
            if (entry) {
                bibliography += `<div id="ref-${key}" class="reference-entry">\n\n`;
                bibliography += this.formatBibliographyEntry(entry);
                bibliography += '\n\n</div>\n\n';
            }
        }

        return bibliography;
    }

    formatBibliographyEntry(entry) {
        let citation = `${entry.author} (${entry.year}). *${entry.title}*.`;
        
        if (entry.journal) {
            citation += ` ${entry.journal}.`;
        } else if (entry.publisher) {
            citation += ` ${entry.publisher}.`;
        }

        if (entry.doi) {
            citation += ` https://doi.org/${entry.doi}`;
        } else if (entry.url) {
            citation += ` ${entry.url}`;
        }

        return citation;
    }
}

// Usage example (for Node.js environment)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CitationProcessor;
}
