function addTask() {
  let task = document.getElementById('newTask').value;
  let li = document.createElement('li');
  li.textContent = task;
  document.getElementById('taskList').appendChild(li);
  document.getElementById('newTask').value = '';
}