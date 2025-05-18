document.addEventListener('DOMContentLoaded', () => {
    const checkBtn = document.getElementById('checkBtn');
    const lastCheck = document.getElementById('lastCheck');
    const resultCard = document.getElementById('resultCard');
    const statsCard = document.getElementById('statsCard');
    const factCard = document.getElementById('factCard');
    const resultLabel = document.getElementById('resultLabel');
    const newsText = document.getElementById('newsText');
  
    // Инициализация Chart.js
    const ctx = document.getElementById('statsChart').getContext('2d');
    const statsChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Real', 'Fake'],
        datasets: [{ data: [0, 0], backgroundColor: ['#4ade80', '#f87171'] }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { callback: value => value + '%' }
          }
        },
        plugins: { legend: { display: false } }
      }
    });
  
    // Скрываем карточки до проверки
    [resultCard, statsCard, factCard].forEach(el => el.classList.add('opacity-0'));
  
    // Обработка клика на примерах
    document.querySelectorAll('.example-link').forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        newsText.value = link.textContent;
        [resultCard, statsCard, factCard].forEach(el => el.classList.add('opacity-0'));
        statsChart.data.datasets[0].data = [0, 0];
        statsChart.update();
      });
    });
  
    // Проверка новости
    checkBtn.addEventListener('click', () => {
      const text = newsText.value.trim();
      if (!text) return;
  
      // Показываем состояние загрузки
      checkBtn.disabled = true;
      const originalText = checkBtn.textContent;
      checkBtn.textContent = 'Loading...';
  
      const form = new FormData();
      form.append('text', text);
      fetch('/predict', { method: 'POST', body: form })
        .then(resp => resp.json())
        .then(data => {
          // Обновляем график
          statsChart.data.datasets[0].data = [data.probs.REAL, data.probs.FAKE];
          statsChart.update();
  
          // Показ результата
          resultLabel.textContent = data.label;
          resultLabel.className = data.label === 'REAL'
            ? 'text-4xl font-bold text-green-600'
            : 'text-4xl font-bold text-red-600';
          resultCard.classList.remove('opacity-0');
          statsCard.classList.remove('opacity-0');
          lastCheck.textContent = 'Last check: ' + new Date().toLocaleTimeString();
  
          // Ссылки fact-check
          const list = document.getElementById('factLinks');
          list.innerHTML = '';
          if (data.links) {
            data.links.forEach(url => {
              const li = document.createElement('li');
              li.innerHTML = `<a href="${url}" target="_blank" class="text-blue-600 hover:underline">${url}</a>`;
              list.appendChild(li);
            });
            factCard.classList.remove('opacity-0');
          }
        })
        .finally(() => {
          checkBtn.disabled = false;
          checkBtn.textContent = originalText;
        });
    });
  });
   