        // Vocabulary data and functionality
        const DEBUG = false;
        const debugLog = (...args) => { if (DEBUG) console.debug('[TSL-ONE]', ...args); };
        let vocabularyData = {};
        let filteredData = {};

        // Category mappings based on ID ranges
        const categoryMappings = {
            countries: { start: 0, end: 11, name: "Countries" },
            landmarks: { start: 12, end: 48, name: "Special locations" },
            numbers: { start: 49, end: 59, name: "Numbers" },
            alphabet: { start: 60, end: 108, name: "Thai Alphabet & Vowels" },
            family: { start: 109, end: 136, name: "Family & Relationships" },
            genders: { start: 137, end: 145, name: "Genders" },
            sports: { start: 154, end: 171, name: "Sports" },
            groceries: { start: 172, end: 183, name: "Groceries" }
        };

        // Debounce helper
        function debounce(fn, wait = 200) {
            let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); };
        }

        // Update results count helper
        function updateResultsCount(count) {
            const el = document.getElementById('resultsCount');
            if (el) el.textContent = `${count} results`;
        }

        // Load vocabulary data
        async function loadVocabulary() {
            try {
                const response = await fetch('tsl_one_s_id_to_label.json');
                vocabularyData = await response.json();
                filteredData = vocabularyData;
                renderVocabulary();
                setupEventListeners();
            } catch (error) {
                console.error('Error loading vocabulary:', error);
                document.getElementById('vocabularyGrid').innerHTML = 
                    '<p style="text-align: center; color: var(--text-medium);">Error loading vocabulary data.</p>';
            }
        }

        // Get category for a given ID
        function getCategory(id) {
            const numId = parseInt(id);
            for (const [category, range] of Object.entries(categoryMappings)) {
                if (numId >= range.start && numId <= range.end) {
                    return category;
                }
            }
            return 'other';
        }

        // Render vocabulary items
        function renderVocabulary() {
            const grid = document.getElementById('vocabularyGrid');
            const items = Object.entries(filteredData)
                .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
                .map(([id, label]) => `
                    <div class="vocabulary-item" data-category="${getCategory(id)}" data-label="${label.toLowerCase()}">
                        <span class="vocabulary-id">${id}</span>
                        <span class="vocabulary-label">${label}</span>
                    </div>
                `).join('');
            
            grid.innerHTML = items || '<p style="text-align: center; color: var(--text-medium);">No items found.</p>';
            updateResultsCount(Object.keys(filteredData).length);
        }

        // Filter by category
        function filterByCategory(category) {
            if (category === 'all') {
                filteredData = vocabularyData;
            } else {
                const range = categoryMappings[category];
                filteredData = {};
                for (let i = range.start; i <= range.end; i++) {
                    if (vocabularyData[i.toString()]) {
                        filteredData[i.toString()] = vocabularyData[i.toString()];
                    }
                }
            }
            
            // Apply current search term if any
            const searchTerm = document.getElementById('vocabularySearch').value;
            if (searchTerm) {
                searchVocabulary(searchTerm);
            } else {
                renderVocabulary();
            }
        }

        // Search vocabulary
        function searchVocabulary(searchTerm) {
            if (!searchTerm) {
                renderVocabulary();
                return;
            }

            const searchResults = {};
            const term = searchTerm.toLowerCase();
            
            for (const [id, label] of Object.entries(filteredData)) {
                if (label.toLowerCase().includes(term) || id === term) {
                    searchResults[id] = label;
                }
            }
            
            const originalFiltered = filteredData;
            filteredData = searchResults;
            renderVocabulary();
            filteredData = originalFiltered;
        }

        // Setup event listeners
        function setupEventListeners() {
            // Search input
            const searchInput = document.getElementById('vocabularySearch');
            searchInput.addEventListener('input', debounce((e) => {
                searchVocabulary(e.target.value);
            }, 200));

            // Category filter buttons
            const filterButtons = document.querySelectorAll('.filter-btn');
            filterButtons.forEach(btn => btn.setAttribute('aria-pressed', btn.classList.contains('active')));
            filterButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    // Update active state
                    filterButtons.forEach(btn => { btn.classList.remove('active'); btn.setAttribute('aria-pressed', 'false'); });
                    e.target.classList.add('active');
                    e.target.setAttribute('aria-pressed', 'true');
                    
                    // Filter vocabulary
                    const category = e.target.getAttribute('data-category');
                    filterByCategory(category);
                    
                    // Clear search input
                    searchInput.value = '';
                });
            });

            // Dark mode removed

            // Form spam-throttle
            (function formTiming(){
                const form = document.querySelector('form[action*="formspree.io"]');
                if (!form) return;
                const ts = document.getElementById('formTimestamp');
                if (ts) ts.value = String(Date.now());
                form.addEventListener('submit', (ev) => {
                    const hp = document.getElementById('hpField');
                    if (hp && hp.value) { ev.preventDefault(); return; }
                    if (ts && ts.value) {
                        const elapsed = Date.now() - Number(ts.value);
                        if (elapsed < 3000) { ev.preventDefault(); return; }
                    }
                });
            })();
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', loadVocabulary);