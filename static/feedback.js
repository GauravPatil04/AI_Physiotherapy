const stars = [1, 2, 3, 4, 5].map((i) => document.getElementById('star-' + i));
    const ratingInput = document.getElementById('rating-input');

    function paintStars(value) {
        stars.forEach((star, idx) => {
            if (idx < value) {
                star.classList.remove('text-zinc-600');
                star.classList.add('text-yellow-400');
            } else {
                star.classList.remove('text-yellow-400');
                star.classList.add('text-zinc-600');
            }
        });
    }

    stars.forEach((star, index) => {
        star.addEventListener('click', () => {
            const selected = index + 1;
            ratingInput.value = String(selected);
            paintStars(selected);
        });
    });

    paintStars(Number(ratingInput.value || 0));
