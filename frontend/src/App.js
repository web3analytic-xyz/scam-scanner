import HomePage from './components/Home';
import ReactGA from 'react-ga';

ReactGA.initialize('UA-252617974-1');

function App() {
  return (
    <div className="App">
      <HomePage />
    </div>
  );
}

export default App;
