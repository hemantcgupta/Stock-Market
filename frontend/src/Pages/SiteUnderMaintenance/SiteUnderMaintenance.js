import React, { Component } from 'react';
import {images} from '../../Constants/images';
import './SiteUnderMaintenance.scss'

class PageNotFound extends Component {
    constructor(props) {
        super(props);
        this.state = {
        }
    }


    render() {
        return (
            <div className='maintenance'>
                {/* <h1>Site is under maintenance</h1> */}
                <img src={images.under_maintenance} alt=''/>
            </div>
        );
    }

}

export default PageNotFound;