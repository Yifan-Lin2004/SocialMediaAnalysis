// switch.js — 点击侧边栏链接，切换可见 section
document.addEventListener('DOMContentLoaded', () => {
    const links   = document.querySelectorAll('nav a[data-target]');
    const sections= document.querySelectorAll('main section[data-page]');
  
    function show(page){
      sections.forEach(sec=>{
        sec.classList.toggle('d-none', sec.dataset.page!==page);
      });
      links.forEach(a=>{
        a.classList.toggle('active', a.dataset.target===page);
      });
    }
  
    links.forEach(a=>{
      a.addEventListener('click', e=>{
        e.preventDefault();
        show(a.dataset.target);
      });
    });
  
    // 初始显示首页
    show('home');
  });
  