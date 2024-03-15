import React, { Component } from 'react';
import './PageNotFound.scss'

class PageNotFound extends Component {
    constructor(props) {
        super(props);
        this.state = {
        }
    }


    render() {
        return (
            <div className='page-not-found'>
                <h1>404 Page Not Found</h1>
            </div>
        );
    }

}

export default PageNotFound;