// Theme toggle functionality
        const themeToggle = document.querySelector('.theme-toggle');
        let isDark = true;

        themeToggle.addEventListener('click', () => {
            isDark = !isDark;
            themeToggle.textContent = isDark ? '☀️' : '🌙';
            // You can add theme switching logic here
        });

        // Upload card click handler
        const uploadCard = document.querySelector('.upload-card');
        uploadCard.addEventListener('click', () => {
            alert('File upload functionality would be implemented here!');
        });

        // Sample button handler
        const sampleButton = document.querySelector('.sample-button');
        sampleButton.addEventListener('click', (e) => {
            e.stopPropagation();
            alert('Loading sample document...');
        });

        // Feature cards hover effect
        const featureCards = document.querySelectorAll('.feature-card');
        featureCards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.boxShadow = '0 8px 32px rgba(163, 113, 247, 0.2)';
            });
            card.addEventListener('mouseleave', () => {
                card.style.boxShadow = 'none';
            });
        });