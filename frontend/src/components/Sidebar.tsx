const Sidebar = () => {
  return (
    <aside className="w-64 flex-shrink-0">
      {/* Sidebar content goes here */}
      <div className="p-4">
        <h2 className="text-lg font-semibold">Sidebar</h2>
        <nav className="mt-4">
          <ul>
            <li>
              <a href="#" className="block py-2">
                Navigation Link 1
              </a>
            </li>
            <li>
              <a href="#" className="block py-2">
                Navigation Link 2
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </aside>
  );
};

export default Sidebar;
