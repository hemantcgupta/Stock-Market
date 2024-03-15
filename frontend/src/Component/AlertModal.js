import React, { Component } from 'react';
import Modal from 'react-bootstrap-modal';
import AlertCard from './AlertCard';
import "./component.scss";


class AlertModal extends Component {
    constructor(props) {
        super(props);
        this.state = {
        }
    }

    componentDidMount() {
        // this.getCardsData()
    }

    render() {
        const { alertData } = this.props;
        return (
            <div>
                <Modal
                    show={this.props.alertVisible}
                    onHide={this.props.closeAlertModal}
                    aria-labelledby="ModalHeader"
                >
                    <Modal.Header closeButton>
                        <Modal.Title id='ModalHeader'>{`Alerts`}</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <div className={`${alertData.length > 0 ? 'cards-modal' : ''}`}>
                            {alertData.length > 0 ? <AlertCard data={this.props.alertData} modal={true} /> : <h4 style={{ textAlign: 'center', color: 'black' }}>No Alerts to Show</h4>}
                        </div>
                    </Modal.Body>
                </Modal>
            </div >
        );
    }
}
export default AlertModal;




