/* ============================================
   DR REJU SAM JOHN — PERSONAL WEBSITE
   JavaScript: Particles, Typing, Animations
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {

    // --- Particle Background ---
    const canvas = document.getElementById('particleCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let particles = [];
        let animationId;

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        class Particle {
            constructor() {
                this.reset();
            }

            reset() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.size = Math.random() * 2 + 0.5;
                this.speedX = (Math.random() - 0.5) * 0.4;
                this.speedY = (Math.random() - 0.5) * 0.4;
                this.opacity = Math.random() * 0.4 + 0.1;
            }

            update() {
                this.x += this.speedX;
                this.y += this.speedY;

                if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
                if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(6, 182, 212, ${this.opacity})`;
                ctx.fill();
            }
        }

        // Create particles (fewer on mobile)
        const isMobile = window.innerWidth < 768;
        const particleCount = isMobile ? 40 : 80;

        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }

        function connectParticles() {
            const maxDist = isMobile ? 100 : 150;
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < maxDist) {
                        const opacity = (1 - dist / maxDist) * 0.15;
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(6, 182, 212, ${opacity})`;
                        ctx.lineWidth = 0.5;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }
        }

        function animateParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                p.update();
                p.draw();
            });
            connectParticles();
            animationId = requestAnimationFrame(animateParticles);
        }

        // Use IntersectionObserver to only animate when hero is visible
        const heroSection = document.getElementById('hero');
        const heroObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateParticles();
                } else {
                    cancelAnimationFrame(animationId);
                }
            });
        }, { threshold: 0.1 });

        heroObserver.observe(heroSection);
    }


    // --- Typing Effect ---
    const typedElement = document.getElementById('typedText');
    if (typedElement) {
        const phrases = [
            'machine learning.',
            'predictive analytics.',
            'intelligent data products.',
            'end-to-end ML pipelines.',
            'scalable data engineering.',
            'cross-functional collaboration.'
        ];

        let phraseIndex = 0;
        let charIndex = 0;
        let isDeleting = false;
        let typingSpeed = 60;

        function typeEffect() {
            const currentPhrase = phrases[phraseIndex];

            if (isDeleting) {
                typedElement.textContent = currentPhrase.substring(0, charIndex - 1);
                charIndex--;
                typingSpeed = 30;
            } else {
                typedElement.textContent = currentPhrase.substring(0, charIndex + 1);
                charIndex++;
                typingSpeed = 60;
            }

            if (!isDeleting && charIndex === currentPhrase.length) {
                typingSpeed = 2000; // Pause at end
                isDeleting = true;
            } else if (isDeleting && charIndex === 0) {
                isDeleting = false;
                phraseIndex = (phraseIndex + 1) % phrases.length;
                typingSpeed = 400; // Pause before next phrase
            }

            setTimeout(typeEffect, typingSpeed);
        }

        setTimeout(typeEffect, 1000);
    }


    // --- Hero Chat Widget ---
    // Production: live FastAPI backend on Render (free tier — first call after
    // ~15min idle is slow due to instance spin-down).
    // For local dev, swap to: 'http://127.0.0.1:8000'
    const RAG_API_URL = 'https://rag-publications-api.onrender.com';

    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');
    const chatBody = document.getElementById('chatBody');
    const chatMessages = document.getElementById('chatMessages');
    const chatWelcome = document.getElementById('chatWelcome');
    const chatSuggestions = document.getElementById('chatSuggestions');

    function appendMessage(role, text) {
        chatWelcome.style.display = 'none';
        const div = document.createElement('div');
        div.className = 'chat-msg ' + role;
        div.textContent = text;
        chatMessages.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
        return div;
    }

    function showTyping() {
        const div = document.createElement('div');
        div.className = 'chat-typing';
        div.id = 'chatTyping';
        div.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function removeTyping() {
        const el = document.getElementById('chatTyping');
        if (el) el.remove();
    }

    function showWakeupNotice() {
        if (document.getElementById('chatWakeupNotice')) return;
        const div = document.createElement('div');
        div.className = 'chat-wakeup';
        div.id = 'chatWakeupNotice';
        div.innerHTML =
            '<div class="wakeup-progress"><div class="wakeup-progress-fill"></div></div>' +
            '<p class="wakeup-text"><i class="fas fa-circle-notch fa-spin"></i>' +
            'Research server warming up — first requests can take up to a minute.</p>';
        chatMessages.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function removeWakeupNotice() {
        const el = document.getElementById('chatWakeupNotice');
        if (el) el.remove();
    }

    async function askQuestion(question) {
        appendMessage('user', question);
        chatSuggestions.style.display = 'none';
        chatSend.disabled = true;
        chatInput.value = '';
        showTyping();

        const wakeupTimer = setTimeout(showWakeupNotice, 4000);
        const controller = new AbortController();
        const abortTimer = setTimeout(() => controller.abort(), 90000);

        try {
            const res = await fetch(RAG_API_URL + '/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question }),
                signal: controller.signal,
            });

            removeTyping();
            removeWakeupNotice();

            if (!res.ok) throw new Error('API returned ' + res.status);

            const data = await res.json();
            appendMessage('assistant', data.answer);
        } catch (err) {
            removeTyping();
            removeWakeupNotice();
            const errDiv = document.createElement('div');
            errDiv.className = 'chat-error';
            errDiv.textContent = err.name === 'AbortError'
                ? 'The server took too long to respond — please try your question again.'
                : 'Service is waking up — please try again in 30s.';
            chatMessages.appendChild(errDiv);
            chatBody.scrollTop = chatBody.scrollHeight;
        } finally {
            clearTimeout(wakeupTimer);
            clearTimeout(abortTimer);
            chatSend.disabled = false;
            chatInput.focus();
        }
    }

    if (chatForm) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const q = chatInput.value.trim();
            if (q) askQuestion(q);
        });
    }

    // Suggestion pills
    document.querySelectorAll('.suggestion-pill').forEach(btn => {
        btn.addEventListener('click', () => {
            askQuestion(btn.getAttribute('data-q'));
        });
    });

    // How-it-works toggle
    const howtoToggle = document.getElementById('howtoToggle');
    const chatHowto = document.getElementById('chatHowto');
    if (howtoToggle && chatHowto) {
        howtoToggle.addEventListener('click', () => {
            const isOpen = howtoToggle.getAttribute('aria-expanded') === 'true';
            howtoToggle.setAttribute('aria-expanded', String(!isOpen));
            chatHowto.hidden = isOpen;
        });
    }


    // --- Scroll Reveal ---
    const revealElements = document.querySelectorAll('.reveal');

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                revealObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    revealElements.forEach(el => revealObserver.observe(el));


    // --- Navbar Scroll Effect ---
    const navbar = document.getElementById('navbar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        lastScroll = currentScroll;
    }, { passive: true });


    // --- Active Nav Link ---
    const sections = document.querySelectorAll('section[id]');
    const navLinksAll = document.querySelectorAll('.nav-links a[href^="#"]');

    function updateActiveLink() {
        const scrollPos = window.pageYOffset + 120;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                navLinksAll.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.style.color = '#06b6d4';
                    } else {
                        link.style.color = '';
                    }
                });
            }
        });
    }

    window.addEventListener('scroll', updateActiveLink, { passive: true });


    // --- Mobile Navigation ---
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navToggle.classList.toggle('active');
            navLinks.classList.toggle('active');
        });

        // Close menu when a link is clicked
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            });
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            }
        });
    }


    // --- Smooth Scroll for Anchor Links ---
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });


    // --- Carousel + Filter ---
    const filterBtns = document.querySelectorAll('.filter-btn');
    const allSlides = Array.from(document.querySelectorAll('.carousel-slide'));
    const carouselTrack = document.querySelector('.carousel-track');
    const carouselPrevBtn = document.getElementById('carouselPrev');
    const carouselNextBtn = document.getElementById('carouselNext');
    const carouselDotsContainer = document.getElementById('carouselDots');
    const carouselCounter = document.getElementById('carouselCounter');

    let visibleSlides = [...allSlides];
    let currentSlide = 0;
    let autoplayTimer = null;

    function buildCarousel() {
        // Reset all slides visibility in DOM
        allSlides.forEach(s => s.style.display = '');

        // Get the active filter
        const activeFilter = document.querySelector('.filter-btn.active');
        const filter = activeFilter ? activeFilter.getAttribute('data-filter') : 'all';

        // Determine visible slides
        visibleSlides = allSlides.filter(s => {
            if (filter === 'all') return true;
            return s.getAttribute('data-category') === filter;
        });

        // Hide filtered-out slides, show matching ones
        allSlides.forEach(s => {
            const matches = filter === 'all' || s.getAttribute('data-category') === filter;
            s.style.display = matches ? '' : 'none';
        });

        // Rebuild track: reorder so visible slides come first
        visibleSlides.forEach(s => carouselTrack.appendChild(s));
        allSlides.filter(s => !visibleSlides.includes(s)).forEach(s => carouselTrack.appendChild(s));

        currentSlide = 0;
        updateCarousel(false);
        buildDots();
    }

    function updateCarousel(animate) {
        if (animate === undefined) animate = true;
        carouselTrack.style.transition = animate ? 'transform 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)' : 'none';
        carouselTrack.style.transform = `translateX(-${currentSlide * 100}%)`;

        // Update dots
        const dots = carouselDotsContainer.querySelectorAll('.carousel-dot');
        dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === currentSlide);
        });

        // Update counter
        if (carouselCounter) {
            carouselCounter.textContent = `${currentSlide + 1} / ${visibleSlides.length}`;
        }
    }

    function buildDots() {
        carouselDotsContainer.innerHTML = '';
        visibleSlides.forEach((_, i) => {
            const dot = document.createElement('button');
            dot.className = 'carousel-dot' + (i === 0 ? ' active' : '');
            dot.setAttribute('aria-label', `Go to slide ${i + 1}`);
            dot.addEventListener('click', () => {
                currentSlide = i;
                updateCarousel();
                resetAutoplay();
            });
            carouselDotsContainer.appendChild(dot);
        });
    }

    function nextSlide() {
        currentSlide = (currentSlide + 1) % visibleSlides.length;
        updateCarousel();
    }

    function prevSlide() {
        currentSlide = (currentSlide - 1 + visibleSlides.length) % visibleSlides.length;
        updateCarousel();
    }

    function startAutoplay() {
        stopAutoplay();
        autoplayTimer = setInterval(nextSlide, 5000);
    }

    function stopAutoplay() {
        if (autoplayTimer) {
            clearInterval(autoplayTimer);
            autoplayTimer = null;
        }
    }

    function resetAutoplay() {
        stopAutoplay();
        startAutoplay();
    }

    // Navigation buttons
    if (carouselNextBtn) carouselNextBtn.addEventListener('click', () => { nextSlide(); resetAutoplay(); });
    if (carouselPrevBtn) carouselPrevBtn.addEventListener('click', () => { prevSlide(); resetAutoplay(); });

    // Pause autoplay on hover
    const carouselEl = document.getElementById('showcaseCarousel');
    if (carouselEl) {
        carouselEl.addEventListener('mouseenter', stopAutoplay);
        carouselEl.addEventListener('mouseleave', startAutoplay);
    }

    // Filter buttons
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            buildCarousel();
            resetAutoplay();
        });
    });

    // Touch / swipe support
    let touchStartX = 0;
    let touchEndX = 0;
    if (carouselEl) {
        carouselEl.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
        }, { passive: true });
        carouselEl.addEventListener('touchend', (e) => {
            touchEndX = e.changedTouches[0].screenX;
            const diff = touchStartX - touchEndX;
            if (Math.abs(diff) > 50) {
                if (diff > 0) nextSlide();
                else prevSlide();
                resetAutoplay();
            }
        }, { passive: true });
    }

    // Initialise
    if (carouselTrack) {
        buildCarousel();
        startAutoplay();
    }


    // --- Lightbox ---
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightboxImg');
    const lightboxCaption = document.getElementById('lightboxCaption');
    const lightboxClose = document.getElementById('lightboxClose');
    const lightboxPrev = document.getElementById('lightboxPrev');
    const lightboxNext = document.getElementById('lightboxNext');

    function openLightbox(src, alt, caption) {
        if (!lightbox || !lightboxImg) return;
        lightboxImg.src = src;
        lightboxImg.alt = alt;
        lightboxCaption.textContent = caption;
        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
        // Hide nav arrows (lightbox shows single image from carousel)
        if (lightboxPrev) lightboxPrev.style.display = 'none';
        if (lightboxNext) lightboxNext.style.display = 'none';
    }

    function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
        if (lightboxPrev) lightboxPrev.style.display = '';
        if (lightboxNext) lightboxNext.style.display = '';
    }

    // Click on carousel image opens lightbox
    allSlides.forEach(slide => {
        const imgEl = slide.querySelector('.slide-image');
        if (imgEl) {
            imgEl.addEventListener('click', () => {
                const img = slide.querySelector('img');
                const title = slide.querySelector('h3').textContent;
                openLightbox(img.src, img.alt, title);
            });
        }
    });

    if (lightboxClose) lightboxClose.addEventListener('click', closeLightbox);

    if (lightbox) {
        lightbox.addEventListener('click', (e) => {
            if (e.target === lightbox) closeLightbox();
        });
    }

    // Keyboard: Escape closes lightbox, arrows navigate carousel
    document.addEventListener('keydown', (e) => {
        if (lightbox && lightbox.classList.contains('active')) {
            if (e.key === 'Escape') closeLightbox();
            return;
        }
        if (e.key === 'ArrowLeft') { prevSlide(); resetAutoplay(); }
        if (e.key === 'ArrowRight') { nextSlide(); resetAutoplay(); }
    });

});
