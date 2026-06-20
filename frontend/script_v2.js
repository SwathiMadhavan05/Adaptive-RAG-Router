document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('query-form');
    const input = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const emptyState = document.getElementById('empty-state');
    const resultsContainer = document.getElementById('results-container');
    
    const routeBadge = document.getElementById('route-badge');
    const confidenceBadge = document.getElementById('confidence-badge');
    const routeDescription = document.getElementById('route-description');
    const probsList = document.getElementById('probs-list');
    
    const latencyVal = document.getElementById('latency-val');
    const costVal = document.getElementById('cost-val');
    const hopsVal = document.getElementById('hops-val');
    
    const answerText = document.getElementById('answer-text');
    
    const contextSection = document.getElementById('context-section');
    const chunksList = document.getElementById('chunks-list');

    const routeColors = {
        'PARAMETRIC': 'var(--route-parametric)',
        'SIMPLE_RAG': 'var(--route-simple)',
        'MULTI_HOP': 'var(--route-multi)',
        'WEB_SEARCH': 'var(--route-web)'
    };

    const routeDescriptions = {
        'PARAMETRIC': 'Answered from model memory without retrieval.',
        'SIMPLE_RAG': 'A single retrieval pass answered the query.',
        'MULTI_HOP': 'Required reasoning and multiple retrieval passes.',
        'WEB_SEARCH': 'Information pulled from recent web search results.'
    };

    // Configure marked.js with syntax highlighting
    marked.setOptions({
        highlight: function(code, lang) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
        },
        langPrefix: 'hljs language-'
    });

    // 3D Tilt Effect for Metric Cards
    const cards = document.querySelectorAll('.metric-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -10; // Max tilt 10deg
            const rotateY = ((x - centerX) / centerX) * 10;
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = input.value.trim();
        if (!query) return;

        // UI Loading State
        submitBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        emptyState.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        contextSection.classList.add('hidden');

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                throw new Error('API request failed');
            }

            const data = await response.json();
            renderResults(data);
            resultsContainer.classList.remove('hidden');
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to fetch response from the router. Make sure the backend is running.');
        } finally {
            submitBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });

    function renderResults(data) {
        // Route Badge
        routeBadge.textContent = data.route;
        routeBadge.className = `route-label route-${data.route}`;
        confidenceBadge.textContent = `${(data.router_confidence * 100).toFixed(1)}% Confidence`;
        routeDescription.textContent = routeDescriptions[data.route] || '';

        // Probabilities
        probsList.innerHTML = '';
        const sortedProbs = Object.entries(data.router_probs).sort((a, b) => b[1] - a[1]);
        
        sortedProbs.forEach(([route, prob]) => {
            const percent = (prob * 100).toFixed(1);
            const color = routeColors[route] || 'gray';
            
            const html = `
                <div class="prob-item">
                    <div class="prob-header">
                        <span>${route}</span>
                        <span>${percent}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: 0%; background-color: ${color}"></div>
                    </div>
                </div>
            `;
            probsList.insertAdjacentHTML('beforeend', html);
        });

        // Trigger animation for prob bars slightly after rendering
        setTimeout(() => {
            const fills = probsList.querySelectorAll('.prob-bar-fill');
            sortedProbs.forEach(([_, prob], index) => {
                fills[index].style.width = `${prob * 100}%`;
            });
        }, 50);

        // Metrics
        latencyVal.textContent = `${data.latency_seconds.toFixed(2)}s`;
        costVal.textContent = `${data.cost_units}x`;
        hopsVal.textContent = data.hops;

        // Answer (Typewriter + Markdown)
        answerText.innerHTML = '';
        const parsedHtml = marked.parse(data.answer);
        
        // We do a simple typewriter effect by adding a cursor element
        // and appending the parsed HTML gradually. (Note: appending HTML char by char is tricky.
        // We'll reveal the HTML content node by node or use a wrapper).
        
        // Simpler approach for markdown typewriter: 
        answerText.innerHTML = '<span id="typing-content"></span><span class="typewriter-cursor"></span>';
        const typingContent = document.getElementById('typing-content');
        const cursor = answerText.querySelector('.typewriter-cursor');
        
        let index = 0;
        // Text-based typewriter
        const textToType = data.answer;
        
        // If it's a very long answer, speed it up
        const speed = Math.max(5, 30 - Math.floor(textToType.length / 50));
        
        function typeWriter() {
            if (index < textToType.length) {
                // To keep markdown formatting somewhat intact during typing, we can just dump the markdown string 
                // and parse it constantly, but that's expensive.
                // Alternatively, we parse once when done, or just type it out and parse at the end.
                // Let's do typing raw, then parse. Or type the already parsed text (requires traversing DOM).
                // Let's go with the simpler one: type out markdown, then parse it when done.
                typingContent.textContent += textToType.charAt(index);
                index++;
                setTimeout(typeWriter, speed);
            } else {
                cursor.remove();
                answerText.innerHTML = marked.parse(data.answer);
            }
        }
        typeWriter();

        // Chunks
        if (data.chunks_retrieved && data.chunks_retrieved.length > 0) {
            chunksList.innerHTML = '';
            data.chunks_retrieved.forEach(chunk => {
                const sourceText = chunk.source ? `<strong>Source:</strong> ${chunk.source}<br/><br/>` : '';
                const chunkHtml = `<div class="chunk-card">${sourceText}${chunk.text}</div>`;
                chunksList.insertAdjacentHTML('beforeend', chunkHtml);
            });
            contextSection.classList.remove('hidden');
        }
    }
});
