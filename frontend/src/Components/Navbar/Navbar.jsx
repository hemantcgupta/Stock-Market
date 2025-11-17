import React from 'react'
import './Navbar.css'
import logo_light from '../../assets/logo-black.png'
import logo_dark from '../../assets/logo-white.png'
import search_icon_light from '../../assets/search-w.png'
import search_icon_dark from '../../assets/search-b.png'
import toogle_light from '../../assets/night.png'
import toogle_dark from '../../assets/day.png'

const Navbar = ({theme, setTheme}) => {

    const toogle_mode = ()=> {
        theme == 'light' ? setTheme('dark') : setTheme('light');
    }
  return (
    <div className='navbar'>
      <img src={theme == 'light' ? logo_light : logo_dark} alt=""  className='logo'/>
      <div className='search-box'>
        <input type="text" placeholder='Search'/>
        <img src={theme == 'light' ? search_icon_light : search_icon_dark} alt="" />
      </div>
      <ul>
        <li>Home</li>
        <li>Products</li>
        <li>Features</li>
        <li>About</li>
      </ul>
      <img onClick={()=>{toogle_mode()}} src={theme == 'light' ? toogle_light : toogle_dark} alt="" className='toggle-icon'/>
    </div>
  )
}

export default Navbar
