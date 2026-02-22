(function(){
	const slider = document.getElementById('slider');
	const slides = Array.from(slider.querySelectorAll('.slide'));
	const indicatorsRoot = document.getElementById('indicators');
	let idx = slides.findIndex(s=>s.classList.contains('active'));
	if (idx < 0) idx = 0;
	let timer = null;
	const interval = 4000;

	function goTo(n){
		// remove current active slide and indicator
		slides[idx].classList.remove('active');
		const prevInd = indicatorsRoot.querySelector('button.active');
		if (prevInd) prevInd.classList.remove('active');

		// set new index and activate
		idx = (n + slides.length) % slides.length;
		slides[idx].classList.add('active');
		const newInd = indicatorsRoot.querySelector(`button[data-index="${idx}"]`);
		if (newInd) newInd.classList.add('active');
	}

	function next(){ goTo(idx+1); }
	function prev(){ goTo(idx-1); }

	// indicators: use event delegation (no creation loop)
	indicatorsRoot.addEventListener('click', (e)=>{
		const btn = e.target.closest('button');
		if (!btn) return;
		const i = Number(btn.getAttribute('data-index'));
		if (Number.isFinite(i)) { goTo(i); resetTimer(); }
	});

	// controls
	slider.querySelectorAll('[data-action="next"]').forEach(b=>b.addEventListener('click', ()=>{ next(); resetTimer(); }));
	slider.querySelectorAll('[data-action="prev"]').forEach(b=>b.addEventListener('click', ()=>{ prev(); resetTimer(); }));

	// keyboard
	window.addEventListener('keydown', e=>{
		if (e.key==='ArrowRight') { next(); resetTimer(); }
		if (e.key==='ArrowLeft')  { prev(); resetTimer(); }
	});

	// touch (swipe)
	let startX = 0;
	slider.addEventListener('touchstart', e=>{ startX = e.touches[0].clientX; });
	slider.addEventListener('touchend', e=>{
		const dx = (e.changedTouches[0].clientX - startX);
		if (Math.abs(dx) > 40) { if (dx < 0) next(); else prev(); resetTimer(); }
	});

	function startTimer(){
		stopTimer();
		timer = setInterval(next, interval);
	}
	function stopTimer(){ if (timer) { clearInterval(timer); timer = null; } }
	function resetTimer(){ stopTimer(); startTimer(); }

	// pause on hover/focus
	slider.addEventListener('mouseenter', stopTimer);
	slider.addEventListener('mouseleave', startTimer);
	slider.addEventListener('focusin', stopTimer);
	slider.addEventListener('focusout', startTimer);

	startTimer();
})();
