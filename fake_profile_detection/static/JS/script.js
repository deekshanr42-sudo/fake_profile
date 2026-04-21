const API_BASE = window.location.origin;
const GALLERY_TITLES = {
    'correlation_heatmap.png': 'Feature Correlation Heatmap',
    'feature_importance.png': 'Random Forest Feature Importance',
    'confusion_matrix_rf.png': 'Confusion Matrix — Random Forest',
    'confusion_matrix_lr.png': 'Confusion Matrix — Logistic Regression',
    'model_comparison.png': 'Model Performance Comparison',
    'distribution_plots.png': 'Feature Distributions — Fake vs Real',
    'engagement_vs_fake.png': 'Engagement Rate vs Follower/Following Ratio'
};

const el = {
    mobileMenuBtn: document.getElementById('mobileMenuBtn'), mobileMenu: document.getElementById('mobileMenu'),
    predictBtn: document.getElementById('predictBtn'), resultPlaceholder: document.getElementById('resultPlaceholder'),
    resultContent: document.getElementById('resultContent'), resultError: document.getElementById('resultError'),
    resultVerdict: document.getElementById('resultVerdict'), verdictEmoji: document.getElementById('verdictEmoji'),
    verdictLabel: document.getElementById('verdictLabel'), verdictConfidence: document.getElementById('verdictConfidence'),
    probBarReal: document.getElementById('probBarReal'), probBarFake: document.getElementById('probBarFake'),
    probValueReal: document.getElementById('probValueReal'), probValueFake: document.getElementById('probValueFake'),
    flagsSection: document.getElementById('flagsSection'), featuresGrid: document.getElementById('featuresGrid'),
    galleryGrid: document.getElementById('galleryGrid'), galleryEmpty: document.getElementById('galleryEmpty'),
    lightbox: document.getElementById('lightbox'), lightboxImg: document.getElementById('lightboxImg'),
    lightboxCaption: document.getElementById('lightboxCaption'), lightboxClose: document.getElementById('lightboxClose'),
    errorMessage: document.getElementById('errorMessage'), followers: document.getElementById('followers'),
    following: document.getElementById('following'), posts: document.getElementById('posts'),
    engagement: document.getElementById('engagement'), profilePic: document.getElementById('profilePic'),
    bioLength: document.getElementById('bioLength'), accountAge: document.getElementById('accountAge'),
};

el.mobileMenuBtn.addEventListener('click', () => el.mobileMenu.classList.toggle('active'));
document.querySelectorAll('.nav-mobile-menu .nav-link').forEach(l => l.addEventListener('click', () => el.mobileMenu.classList.remove('active')));

const REAL_EXAMPLE = { followers: 5000, following: 800, posts: 500, engagement: 6.5, profilePic: '1', bioLength: 180, accountAge: 730 };
const FAKE_EXAMPLE = { followers: 25, following: 3500, posts: 3, engagement: 0.2, profilePic: '0', bioLength: 5, accountAge: 15 };

function fillForm(data) {
    el.followers.value = data.followers; el.following.value = data.following; el.posts.value = data.posts;
    el.engagement.value = data.engagement; el.profilePic.value = data.profilePic;
    el.bioLength.value = data.bioLength; el.accountAge.value = data.accountAge;
}

document.getElementById('loadReal').addEventListener('click', () => fillForm(REAL_EXAMPLE));
document.getElementById('loadFake').addEventListener('click', () => fillForm(FAKE_EXAMPLE));

el.predictBtn.addEventListener('click', async () => {
    const payload = {
        followers: Number(el.followers.value) || 0, following: Number(el.following.value) || 0,
        posts: Number(el.posts.value) || 0, engagement_rate: Number(el.engagement.value) || 0,
        has_profile_picture: el.profilePic.value === '1', bio_length: Number(el.bioLength.value) || 0,
        account_age_days: Number(el.accountAge.value) || 1
    };
    el.predictBtn.classList.add('loading'); el.predictBtn.disabled = true; showPlaceholder();
    try {
        const res = await fetch(`${API_BASE}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await res.json();
        data.status === 'success' ? displayResult(data) : showError(data.message || 'Unknown error');
    } catch (e) { showError('Could not reach the server. Make sure <code>python server.py</code> is running.'); }
    finally { el.predictBtn.classList.remove('loading'); el.predictBtn.disabled = false; }
});

function showPlaceholder() { el.resultPlaceholder.style.display = 'flex'; el.resultContent.style.display = 'none'; el.resultError.style.display = 'none'; }
function showError(msg) { el.resultPlaceholder.style.display = 'none'; el.resultContent.style.display = 'none'; el.resultError.style.display = 'flex'; el.errorMessage.innerHTML = msg; }

function displayResult(data) {
    el.resultPlaceholder.style.display = 'none'; el.resultError.style.display = 'none'; el.resultContent.style.display = 'block';
    const isFake = data.prediction === 'Fake';
    el.resultVerdict.className = 'result-verdict ' + (isFake ? 'fake-verdict' : 'real-verdict');
    el.verdictEmoji.textContent = isFake ? '🚨' : '✅';
    el.verdictLabel.textContent = isFake ? 'FAKE PROFILE' : 'REAL PROFILE';
    el.verdictConfidence.textContent = `Confidence: ${data.confidence}%`;

    const realPct = (data.probability_real * 100).toFixed(1), fakePct = (data.probability_fake * 100).toFixed(1);
    el.probValueReal.textContent = realPct + '%'; el.probValueFake.textContent = fakePct + '%';
    el.probBarReal.style.width = '0%'; el.probBarFake.style.width = '0%';
    requestAnimationFrame(() => { requestAnimationFrame(() => { el.probBarReal.style.width = realPct + '%'; el.probBarFake.style.width = fakePct + '%'; }); });

    let flagsHTML = '';
    if (data.red_flags?.length) { flagsHTML += '<h4>⚠️ Red Flags Detected</h4>'; data.red_flags.forEach(f => flagsHTML += `<div class="flag-item flag-red"><span class="flag-icon">🔴</span>${f}</div>`); }
    if (data.green_flags?.length) { flagsHTML += '<h4 style="margin-top:12px">✅ Positive Signals</h4>'; data.green_flags.forEach(f => flagsHTML += `<div class="flag-item flag-green"><span class="flag-icon">🟢</span>${f}</div>`); }
    el.flagsSection.innerHTML = flagsHTML;

    if (data.features_used) {
        const labels = { follower_following_ratio: 'F/F Ratio', posts_per_day: 'Posts/Day', bio_completeness: 'Bio Completeness', account_freshness: 'Freshness', interaction_score: 'Interaction Score' };
        let fHTML = '';
        for (const [k, v] of Object.entries(data.features_used)) fHTML += `<div class="feature-item"><span class="feature-name">${labels[k]||k}</span><span class="feature-value">${v}</span></div>`;
        el.featuresGrid.innerHTML = fHTML;
    }
    if (window.innerWidth <= 900) el.resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadGallery() {
    try {
        const res = await fetch(`${API_BASE}/api/visualizations`); const data = await res.json();
        if (!data.files?.length) { el.galleryGrid.style.display = 'none'; el.galleryEmpty.style.display = 'block'; return; }
        el.galleryGrid.style.display = 'grid'; el.galleryEmpty.style.display = 'none';
        el.galleryGrid.innerHTML = data.files.map(f => {
            const t = GALLERY_TITLES[f]||f, u = `${API_BASE}/viz/${encodeURIComponent(f)}`;
            return `<div class="gallery-item" data-img="${u}" data-title="${t}"><img src="${u}" alt="${t}" loading="lazy"><div class="gallery-item-caption">${t}</div></div>`;
        }).join('');
        document.querySelectorAll('.gallery-item').forEach(i => i.addEventListener('click', () => openLightbox(i.dataset.img, i.dataset.title)));
    } catch (e) { el.galleryGrid.style.display = 'none'; el.galleryEmpty.style.display = 'block'; }
}

function openLightbox(src, cap) { el.lightboxImg.src = src; el.lightboxCaption.textContent = cap; el.lightbox.classList.add('active'); document.body.style.overflow = 'hidden'; }
function closeLightbox() { el.lightbox.classList.remove('active'); document.body.style.overflow = ''; }
el.lightboxClose.addEventListener('click', closeLightbox);
el.lightbox.addEventListener('click', e => { if (e.target === el.lightbox) closeLightbox(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });
document.querySelectorAll('a[href^="#"]').forEach(a => a.addEventListener('click', function(e) { e.preventDefault(); document.querySelector(this.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' }); }));
document.addEventListener('DOMContentLoaded', loadGallery);