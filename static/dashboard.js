function toggleDropdown() {
    const dropdown = document.getElementById("profile-dropdown");
    dropdown.classList.toggle("hidden");
}

function openExercise(exerciseName) {
    fetch('/exercise/' + exerciseName)
        .then(() => {
            window.location.href = '/exercise_page';
        });
}

window.onclick = function(event) {
    if (!event.target.closest('.profile-container')) {
        document.getElementById("profile-dropdown").classList.add("hidden");
    }
}
