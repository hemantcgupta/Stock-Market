import React, { Component } from 'react';
import { images } from '.././Constants/images';
import cookieStorage from '.././Constants/cookie-storage';
import config from '.././Constants/config';
import eventApi from '.././API/eventApi';
import { Tooltip } from '@material-ui/core';


const API_URL = config.API_URL

class DownloadCSV extends Component {

    sendEvent(name, path, page) {
        var eventData = {
            event_id: '3',
            description: `User downloaded ${name}`,
            where_path: path,
            page_name: page
        }
        eventApi.sendEvent(eventData)
    }

    render() {
        let userDetails = JSON.parse(cookieStorage.getCookie('userDetails'))
        const { url, name, path, page,changeColor } = this.props;
        return (
            <Tooltip title = {<h4>{name}</h4>}>
            <a className="download"
                href={`${url}&uid=${userDetails.id}&download=do`}
                onClick={() => this.sendEvent(name, path, page)}
                target='-/'>
                <i className='fa fa-download' style={{color: changeColor ? 'black' : 'orange'}}></i>
            </a>
            </Tooltip>
        );
    }
}

export default DownloadCSV;
